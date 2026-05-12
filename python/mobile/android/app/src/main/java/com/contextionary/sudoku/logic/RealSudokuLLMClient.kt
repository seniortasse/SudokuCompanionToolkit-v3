package com.contextionary.sudoku.logic

import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.util.UUID
import com.contextionary.sudoku.telemetry.ConversationTelemetry
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.delay
import java.io.IOException

import com.contextionary.sudoku.conductor.policy.IntentEnvelopeV1

/**
 * Phase 4 — Option A (Universe A -> Universe B):
 * - Telemetry-silent: NO ConversationTelemetry emissions here.
 * - ID-silent: does NOT generate authoritative chain IDs (modelCallId/toolplanId/policyReqSeq/turnId/etc).
 * - Pure network adapter + parsing:
 *   build payload -> call model -> parse -> return results only.
 *
 * NOTE: We keep ModelCallTelemetryCtx because call sites pass it around.
 * RealSudokuLLMClient only uses it to emit MODEL_PAYLOAD_OUT/IN when provided.
 */
data class ModelCallTelemetryCtx(
    val modelCallId: String? = null,
    val turnId: Long? = null,
    val turnSeq: Int? = null,
    val tickId: Int? = null,
    val policyReqSeq: Long? = null,
    val toolplanId: String? = null,
    val correlationId: String? = null,
    val mode: String = "GRID",
    val stateHeaderSha12: String? = null,

    // Phase 8 — reply payload audit context
    val demandCategory: String? = null,
    val rolloutMode: String? = null
)

private data class HttpAttemptResult(
    val ok: Boolean,
    val code: Int,
    val body: String,
    val retryAfterMs: Long? = null
)

class RealSudokuLLMClient(
    private val apiKey: String,
    private val model: String
) : SudokuLLMClient {

    // Network policy:
    // - Fail fast on timeouts (avoid blocking a turn for ~30s)
    // - Retry only when it makes sense (429, some 5xx), with backoff + jitter
    // - Keep timeouts bounded so UI remains responsive
    private val httpClient = OkHttpClient.Builder()
        .callTimeout(12, java.util.concurrent.TimeUnit.SECONDS)
        .connectTimeout(5, java.util.concurrent.TimeUnit.SECONDS)
        .writeTimeout(12, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(12, java.util.concurrent.TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    // -------------------------------------------------------------------------
    // Expert contract block (kept as-is — this is "real job": prompt shaping)
    // -------------------------------------------------------------------------

    private val SUDOKU_EXPERT_BLOCK = """
SUDOKU / GRID MODE — STRICT CONTRACTS (DO NOT VIOLATE)

0) Mission (this drives your behavior; NOT a script)
You must achieve BOTH outcomes efficiently:
- TRUTH: on-screen grid must match the paper/book 100%.
- READINESS: once matched, the grid must be uniquely solvable before solve-assist.

Efficiency target:
- Minimize turns. User can confirm ONE cell per turn.
- Avoid “dead turns”: every GRID_MODE response must advance the mission.

A) Sudoku invariants (always true)
- The grid is 9×9 with exactly 81 cells.
- Digits are 1–9; digit 0 means “blank/empty”.
- Rows r1..r9, columns c1..c9, 3×3 boxes: total houses = 27.

B) Grounding & factuality rules (authoritative sources)
- You will receive a GRID_CONTEXT block from the app.
- Treat CURRENT_DISPLAY_DIGITS_0_MEANS_BLANK as the truth of what is currently shown on screen.
- Treat TRUTH_LAYER_GIVENS as “printed clues” (highest trust, but can still be wrong due to scan error).
- Treat TRUTH_LAYER_USER_SOLUTIONS as “filled answers” that may be wrong unless confirmed.
- Treat CANDIDATES as noisy hints (never treat them as certain facts).

You MUST NOT invent:
- any digit in any row/col,
- any counts (givens/conflicts/unresolved/etc.),
- any solvability claims (“unique solution”, “solvable”, etc.)
unless explicitly provided in GRID_CONTEXT.

Conflict grounding (HARD):
- You may ONLY claim a Sudoku contradiction if it appears in CONFLICTS_DETAILS.
- If CONFLICTS_DETAILS says “(none)”, you MUST NOT claim any contradiction.

C) Output format (hard requirement)
- You MUST respond ONLY using tool calls that match the provided JSON schema.
- Always emit at least one tool call.
- The reply(text=...) tool must be present and reply.text must be non-empty.
- Do NOT output any extra text outside tool calls.

D) Friend + Coach identity (how you should sound)
You are Sudo: 50% friendly companion, 50% practical Sudoku coach.

Friend side:
- Warm, human, supportive. Add light “color” only if relevant to the moment (scan confusion / corrections / user emotion).
- Do NOT derail the task.

Coach side:
- Be efficient, action-driven, and crystal-clear.
- Support every claim with GRID_CONTEXT (or explicitly say you need confirmation).
- Always leave the user with ONE clear next step.

E) COMPLETE TURN CONTRACT (HARD, LLM-FIRST, NO SCRIPTS)
Your replies must be COMPLETE and ACTIONABLE.

Every GRID_MODE reply must include:
1) Human warmth appropriate to the moment (short is okay, but not cold).
2) A clear evidence-backed explanation of the current situation (or explicit uncertainty).
3) ONE specific next action (NON-VAGUE):
   - ask_confirm_cell_rc(...) OR recommend_validate OR recommend_retake
   - or (if needed): confirm_interpretation / ask_clarifying_question / switch_to_tap.
4) Never end with vague closings like “Let’s start by checking one of them together.”
   The user must know exactly what happens next.

If you emit apply_user_edit_rc/apply_user_edit in a response:
1) Confirm what you changed and where (plain language).
2) Explain consequences WITHOUT inventing:
   - If GRID_CONTEXT explicitly provides resulting status (solvability/conflicts/mismatch/unresolved), you may state it.
   - Otherwise speak conditionally (“This should remove that contradiction; next we’ll re-check the grid state.”)
3) Provide ONE next step in the same response (ask_confirm_cell_rc OR recommend_validate OR recommend_retake).

F) Corrections & edits (MANDATORY TOOL EMISSION)
- If you need the user to verify a specific cell: use ask_confirm_cell_rc(row, col, prompt).

- If the user is answering a pending cell-value question (pending_ctx=ask_cell_value with row+col):
  You MUST treat that answer as an implicit edit authorization for that specific cell, and in the SAME response you MUST:
  1) emit confirm_cell_value_rc(row, col, digit_or_blank)
  2) if digit_or_blank differs from CURRENT_DISPLAY at (row,col), emit apply_user_edit_rc(row, col, digit_or_blank, source="user_text" or "user_voice")
  3) if a progress/update tool exists, emit it after apply_user_edit_rc
  You MUST NOT “just say thanks / confirmed” without emitting confirm_cell_value_rc.

- Outside pending:
  Only apply a digit change when the user explicitly requests it:
  -> emit apply_user_edit_rc(row, col, digit, source="user_text" or "user_voice") in the same response.

- Never claim you changed a digit unless you also emitted apply_user_edit_rc (or apply_user_edit legacy) in the same response.

G) CORRECTION / NEXT-CHECK POLICY (STRICT PRIORITY + 4 CASES)
Allowed sets for proposing a next check (HARD):
1) mismatch_indices_vs_deduced (HIGHEST priority; if non-empty)
2) unresolved_indices (ONLY when solvability == "none")
You MUST NOT propose next-check cells from low_confidence_indices or auto_changed_indices.

H) ONE-STEP DISCIPLINE
- Only propose ONE actionable step per turn.
- If you propose a single cell check, you MUST also ask that cell using ask_confirm_cell_rc(row,col,...).
- Never propose a cell outside mismatch_indices_vs_deduced or unresolved_indices (and unresolved only when solvability=="none").

I) Tool usage reminder (GRID MODE)
- Every response MUST include reply(text="...") with non-empty text.
- Use ask_confirm_cell_rc for targeted verification.
- Use ask_user_to_confirm_validation to ask if the on-screen grid matches the paper.
- If the user’s response is unclear, use clarify_validation (do NOT repeat ask_user_to_confirm_validation).
- If the user confirms match while pending:confirm_validate, emit finalize_validation_presentation + start_solving.
- Use recommend_retake when the state cannot be made solve-ready without a better scan.

PLAYER-LANGUAGE / JARGON RULE (HARD)
- The app may provide a fact bundle of type GLOSSARY_BUNDLE.
- You MUST use it to translate internal terms and UI cues into natural Sudoku-player speech.

DO NOT say these internal words/phrases as-is:
- "unresolved", "low confidence", "severity", "conflict scoop", "mismatch scoop", "hiding" (for visible digits)

Preferred phrasing:
- low confidence -> "the scan isn’t sure"
- unresolved -> "needs your confirmation / question-mark cell"
- conflict -> "breaks Sudoku rules right now (duplicate in a row/column/box)"
- mismatch -> "one of your filled-in answers is wrong (compared to the correct solution implied by the clues)"
- autoCorrect/changed -> "I made a safe scan correction (cyan outline)"

COORDINATE SPEECH (HARD)
- Speak as "row X column Y". Use rXcY only if the user asks for coordinates explicitly.

BOX NAMING (HARD)
- Use: top-left, top-middle, top-right, middle-left, center, middle-right, bottom-left, bottom-middle, bottom-right.
- Do not say "box 7" unless you also provide the position name and you have defined the numbering rule once.

PHASE RULE:
                - Use only the 'phase' field to choose your posture:
                  * CONFIRMING: focus on making on-screen grid match paper/book (resolve conflicts, uncertain cells, answer grid questions).
                  * SEALING: summarize clean grid + confirm readiness to start solving (ask for the user's go-ahead).
                  * SOLVING: coach next move using solver/coach evidence; keep the user moving step-by-step.

(unchanged)
""".trimIndent()

    private fun augmentSystemPrompt(base: String): String {
        val b = base.trim()

        val hardOverride = """
HARD OVERRIDE (takes precedence over any earlier instructions)

0) USER-FIRST PRIORITY LADDER (HARD — NEVER VIOLATE)
The user is the driver. ALWAYS respond to what the user just said, even if pending exists.
Pending is context only, never a higher-priority instruction.

If the user asks ANY direct question or requests ANY action (even non-Sudoku, e.g. “what’s the weather in Moscow?”):
- Answer that FIRST using reply(text=...).
- You MAY optionally ask “Want to continue the grid check?” as your ONE next step.

PENDING RESOLUTION (ONLY WHEN THE USER IS CLEARLY ANSWERING IT)
If pending_ctx indicates ask_cell_value for a specific cell (row/col or idx→row/col),
you may treat the user’s message as answering the pending question ONLY if:
- The user does NOT explicitly mention a DIFFERENT row/col, and
- The user’s intent is clearly “this cell is X/blank”.

When you DO treat it as an answer to the pending cell:
- emit confirm_cell_value_rc(row,col,digit_or_blank) in this same response.
- If digit_or_blank differs from CURRENT_DISPLAY at that cell, ALSO emit apply_user_edit_rc(row,col,digit_or_blank,source=...).
- If the digit is not clear, emit ask_clarifying_question (do not guess).

EXPLICIT OVERRIDE OF PENDING (HARD)
If the user explicitly references a different cell than the pending cell (e.g., user says “r3c8 is 5” while pending is r3c9):
- Do NOT resolve the pending cell in that turn.
- Treat the user statement as the new priority:
  - If it is a clear edit/confirmation for that referenced cell: emit confirm_cell_value_rc and (if mismatch) apply_user_edit_rc for THAT cell.
  - Otherwise ask one clarifying question about THAT cell.
- Keep the original pending item pending unless the user explicitly cancels it.

VALIDATION DECISION (LLM-FIRST, HARD)
If STATE_HEADER indicates pending:confirm_validate, you MUST decide what the user meant.

You must choose exactly ONE path:

Path A — User validated (clear confirmation that grid matches paper):
- You MUST emit BOTH:
    - finalize_validation_presentation(reason="validated")
    - start_solving
- reply.text MUST:
    - confirm validation in one sentence
    - announce SOLVING mode in one sentence (what changes: hints/techniques/next moves)

Path B — User did NOT validate OR meaning is unclear:
- do NOT emit start_solving
- If the previous turn already asked for validation (pending:confirm_validate is set):
    - DO NOT repeat ask_user_to_confirm_validation.
    - Emit clarify_validation(reason=ASR_GARBLED|MIXED_SIGNAL|OFF_TOPIC|PARTIAL_CONFIRM, style=YES_NO|SPOT_CHECK_3|ASK_WHICH_CELL_MISMATCHES, prompt="...")
- Else (not yet pending confirm_validate):
    - Emit ask_user_to_confirm_validation (one clear yes/no question in reply.text).

Never rely on literal words like "yes/no". Use context and meaning.
If the user’s message contains both affirmation and doubt, treat it as NOT validated and ask a clarifying question.

TOOLPLAN RATIONALE (INTERNAL META — LOGGING ONLY)
- Include toolplan_rationale as the FIRST tool call in every response.
- toolplan_rationale.summary must be <= 600 characters.
- In summary, briefly mention:
  (a) key grid facts you used (from GRID_CONTEXT),
  (b) the rule(s) you followed from this prompt,
  (c) why you chose the control tool you chose.
- This is NOT user-facing; keep it concise and factual.

1) HARD RULE — Reply must verbalize the control tool
If you emit any of these tools: ask_confirm_cell_rc, confirm_interpretation, ask_clarifying_question, switch_to_tap, recommend_validate, recommend_retake
then reply.text MUST include the same next-step question/instruction in natural language.
reply.text must explicitly include the same row/col for ask_confirm_cell_rc.
reply.text must end with that one question/instruction.

2) NEXT-CHECK ALLOWED SETS (STRICT)
You may ONLY propose/ask a cell-check from:
- mismatch_indices_vs_deduced (highest priority)
- unresolved_indices (only when solvability == "none")
You MUST NOT drive checks using low_confidence_indices or auto_changed_indices.

3) FOUR-CASE POLICY (STRICT)
(unchanged)

4) CONFLICT GROUNDING (STRICT)
- You may ONLY claim a contradiction if it appears in CONFLICTS_DETAILS.
- If CONFLICTS_DETAILS says “(none)”, you MUST NOT claim any contradiction.
""".trimIndent()

        val solvingRules = """
SOLVING MODE (STRICT) — applies when stateHeader contains phase=SOLVING or grid_phase=SOLVING

1) SOLVING mission
- You are now a Sudoku coach: give the next best action / technique based on the solver engine step(s).

2) Truth contract (HARD)
- The engine-provided step is authoritative.
- Do NOT invent a technique/pattern not supported by the engine output.
- If the engine output is missing/empty, ask for refresh_solve_step (do not guess).

3) Tools preference in SOLVING (when available)
- show_solve_overlay(style="full") when user asks: “show me / draw it / make it visual”
- show_solve_overlay(style="mini") when user asks: “quick hint”
- refresh_solve_step when user asks: “another technique / different approach / next step”
- recommend_validate is FORBIDDEN in SOLVING (never propose validate talk again)
""".trimIndent()

        return buildString {
            appendLine(b)
            appendLine()
            appendLine("-----")
            appendLine(SUDOKU_EXPERT_BLOCK)
            appendLine()
            appendLine("-----")
            appendLine(hardOverride)
            appendLine()
            appendLine(solvingRules)
        }.trim()
    }

    // -------------------------------------------------------------------------
    // Helpers (pure parsing / shaping, no telemetry)
    // -------------------------------------------------------------------------

    private fun parseRetryAfterMs(h: String?): Long? {
        val s = h?.trim().orEmpty()
        if (s.isEmpty()) return null
        val seconds = s.toLongOrNull() ?: return null
        return (seconds * 1000L).coerceAtMost(15_000L)
    }

    /**
     * Executes request with:
     * - FAIL FAST on SocketTimeoutException (NO retries)
     * - Retry with exponential backoff on 429 (honor Retry-After if present)
     * - Optional short retry on 5xx
     */
    private suspend fun executeWithRetryPolicy(
        request: Request,
        reqId: String,
        tag: String,
        payloadSha12: String,
        maxRetries429: Int = 2,
        maxRetries5xx: Int = 1
    ): HttpAttemptResult {

        var attempt429 = 0
        var attempt5xx = 0
        var attemptIo = 0

        while (true) {
            var call: okhttp3.Call? = null
            val ctx = currentCoroutineContext()
            val ctxActiveAtAttemptStart = ctx.isActive

            try {
                if (!ctxActiveAtAttemptStart) {
                    Log.w("SudokuLLM", "$tag coroutine NOT active before call start req_id=$reqId payload_sha12=$payloadSha12")
                    return HttpAttemptResult(ok = false, code = -3, body = "cancelled")
                }

                call = httpClient.newCall(request)

                Log.i("SudokuLLM", "$tag call_start req_id=$reqId payload_sha12=$payloadSha12 ctxActive=$ctxActiveAtAttemptStart")

                call.execute().use { resp ->
                    val code = resp.code
                    val body = resp.body?.string().orEmpty()

                    if (resp.isSuccessful) {
                        return HttpAttemptResult(ok = true, code = code, body = body)
                    }

                    if (code == 429 && attempt429 < maxRetries429) {
                        attempt429++
                        val retryAfterMs = parseRetryAfterMs(resp.header("Retry-After"))
                        val base = retryAfterMs ?: (400L shl (attempt429 - 1))
                        val jitter = kotlin.random.Random.nextLong(0L, 250L)
                        val delayMs = (base + jitter).coerceAtMost(2500L)

                        Log.w("SudokuLLM", "$tag HTTP 429 retry=$attempt429 req_id=$reqId delayMs=$delayMs payload_sha12=$payloadSha12")
                        delay(delayMs)
                        return@use
                    }

                    if (code in 500..599 && attempt5xx < maxRetries5xx) {
                        attempt5xx++
                        val base = 250L shl (attempt5xx - 1)
                        val jitter = kotlin.random.Random.nextLong(0L, 200L)
                        val delayMs = (base + jitter).coerceAtMost(1200L)

                        Log.w("SudokuLLM", "$tag HTTP $code retry=$attempt5xx req_id=$reqId delayMs=$delayMs payload_sha12=$payloadSha12")
                        delay(delayMs)
                        return@use
                    }

                    return HttpAttemptResult(ok = false, code = code, body = body)
                }

                continue

            } catch (e: CancellationException) {
                val canceledFlag = (call?.isCanceled() == true)
                Log.w(
                    "SudokuLLM",
                    "$tag coroutine CancellationException req_id=$reqId ctxActive=${currentCoroutineContext().isActive} " +
                            "callExists=${call != null} callCanceledFlag=$canceledFlag payload_sha12=$payloadSha12",
                    e
                )
                try { call?.cancel() } catch (_: Exception) {}
                return HttpAttemptResult(ok = false, code = -3, body = "cancelled")

            } catch (e: java.net.SocketTimeoutException) {
                Log.e("SudokuLLM", "$tag timeout FAIL-FAST req_id=$reqId payload_sha12=$payloadSha12", e)
                return HttpAttemptResult(ok = false, code = -1, body = "timeout")

            } catch (e: java.io.InterruptedIOException) {
                val canceledFlag = (call?.isCanceled() == true)
                val ctxActiveNow = currentCoroutineContext().isActive

                Log.w(
                    "SudokuLLM",
                    "$tag InterruptedIOException req_id=$reqId ctxActive=$ctxActiveNow canceledFlag=$canceledFlag " +
                            "msg='${e.message}' cause='${e.cause?.message}' payload_sha12=$payloadSha12",
                    e
                )

                if (canceledFlag || !ctxActiveNow) {
                    Log.w("SudokuLLM", "$tag cancelled (InterruptedIOException) req_id=$reqId payload_sha12=$payloadSha12")
                    return HttpAttemptResult(ok = false, code = -3, body = "cancelled")
                }

                Log.w("SudokuLLM", "$tag interrupted IO (treat as timeout) req_id=$reqId payload_sha12=$payloadSha12", e)
                return HttpAttemptResult(ok = false, code = -1, body = "timeout")

            } catch (e: IOException) {
                val canceledFlag = (call?.isCanceled() == true)
                val ctxActiveNow = currentCoroutineContext().isActive

                Log.w(
                    "SudokuLLM",
                    "$tag IOException req_id=$reqId ctxActive=$ctxActiveNow canceledFlag=$canceledFlag " +
                            "msg='${e.message}' cause='${e.cause?.message}' payload_sha12=$payloadSha12",
                    e
                )

                if (canceledFlag || !ctxActiveNow) {
                    Log.w("SudokuLLM", "$tag cancelled (IOException) req_id=$reqId payload_sha12=$payloadSha12")
                    return HttpAttemptResult(ok = false, code = -3, body = "cancelled")
                }

                if (attemptIo < 1) {
                    attemptIo++
                    val delayMs = 250L + kotlin.random.Random.nextLong(0L, 200L)
                    Log.w("SudokuLLM", "$tag IO exception retry=$attemptIo req_id=$reqId delayMs=$delayMs payload_sha12=$payloadSha12", e)
                    delay(delayMs)
                    continue
                }

                Log.e("SudokuLLM", "$tag IO exception giving up req_id=$reqId payload_sha12=$payloadSha12", e)
                return HttpAttemptResult(ok = false, code = -2, body = "io_error:${e.javaClass.simpleName}")
            }
        }
    }

    private fun capPreview(text: String, maxChars: Int): String {
        val normalized = text
            .replace('\n', ' ')
            .replace('\r', ' ')
            .replace(Regex("\\s+"), " ")
            .trim()
        if (normalized.length <= maxChars) return normalized
        return normalized.substring(0, maxChars).trimEnd() + "…"
    }

    private fun sha256HexUtf8(s: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(s.toByteArray(Charsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        for (b in digest) hex.append(String.format("%02x", b))
        return hex.toString()
    }

    // ---- Payload logging gate ----
    // We now always emit full prompt/response text into telemetry (with a generous clip cap)
    // so audit tools can reconstruct the exact System / Developer / User prompt surface.
    private val FULL_TELEMETRY_TEXT_MAX_CHARS: Int = 250_000

    private fun clipForTelemetry(s: String, max: Int = FULL_TELEMETRY_TEXT_MAX_CHARS): String {
        if (s.length <= max) return s
        return s.substring(0, max) + "…(truncated)"
    }

    private fun payloadKindForChannel(channel: String): String =
        when (channel.uppercase()) {
            "INTENT" -> "meaning_extract_v1"
            "REPLY" -> "reply_generate_v1"
            "FREE_TALK" -> "free_talk_v1"
            else -> "model_call_v1"
        }

    private fun emitModelPayloadOut(
        ctx: ModelCallTelemetryCtx?,
        channel: String,                 // "INTENT" | "REPLY" | ...
        payloadStr: String,
        reqId: String? = null,
        payloadSha12: String? = null,
        systemPrompt: String? = null,
        developerPrompt: String? = null,
        userMessage: String? = null
    ) {
        if (ctx == null) return

        val sha12 = payloadSha12 ?: runCatching { sha256HexUtf8(payloadStr).take(12) }.getOrNull()
        val payloadKind = payloadKindForChannel(channel)

        ConversationTelemetry.emit(
            mapOf(
                "type" to "MODEL_PAYLOAD_OUT",
                "channel" to channel,
                "payload_kind" to payloadKind,

                "turn_id" to ctx.turnId,
                "tick_id" to ctx.tickId,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,

                "req_id" to reqId,

                "payload_len" to payloadStr.length,
                "payload_sha12" to sha12,

                // Full request body for audit reconstruction
                "payload_text" to clipForTelemetry(payloadStr),
                "payload_preview" to capPreview(payloadStr, 900),

                // First-class prompt surface fields for audit readability
                "system_prompt_text" to systemPrompt?.let { clipForTelemetry(it) },
                "developer_prompt_text" to developerPrompt?.let { clipForTelemetry(it) },
                "user_message_text" to userMessage?.let { clipForTelemetry(it) },

                "system_prompt_len" to systemPrompt?.length,
                "developer_prompt_len" to developerPrompt?.length,
                "user_message_len" to userMessage?.length,

                "system_prompt_sha12" to systemPrompt?.let { sha256HexUtf8(it).take(12) },
                "developer_prompt_sha12" to developerPrompt?.let { sha256HexUtf8(it).take(12) },
                "user_message_sha12" to userMessage?.let { sha256HexUtf8(it).take(12) }
            )
        )
    }

    private fun emitModelPayloadIn(
        ctx: ModelCallTelemetryCtx?,
        channel: String,
        responseStr: String,
        reqId: String? = null,
        httpCode: Int? = null,
        dtMs: Long? = null,
        payloadSha12: String? = null,
        parseOk: Boolean? = null,
        parseErrors: String? = null
    ) {
        if (ctx == null) return

        val respSha12 = runCatching { sha256HexUtf8(responseStr).take(12) }.getOrNull()
        val payloadKind = payloadKindForChannel(channel)

        ConversationTelemetry.emit(
            mapOf(
                "type" to "MODEL_PAYLOAD_IN",
                "channel" to channel,
                "payload_kind" to payloadKind,

                "turn_id" to ctx.turnId,
                "tick_id" to ctx.tickId,
                "policy_req_seq" to ctx.policyReqSeq,
                "correlation_id" to ctx.correlationId,
                "model_call_id" to ctx.modelCallId,
                "toolplan_id" to ctx.toolplanId,

                "req_id" to reqId,
                "http_code" to httpCode,
                "dt_ms" to dtMs,
                "payload_sha12" to payloadSha12,

                "response_len" to responseStr.length,
                "response_sha12" to respSha12,

                "response_text" to clipForTelemetry(responseStr),
                "response_preview" to capPreview(responseStr, 1200),

                "parse_ok" to parseOk,
                "parse_errors" to parseErrors
            )
        )
    }

    // -------------------------------------------------------------------------
    // Generic helper for minimal JSON calls: IntentEnvelope + ReplyGenerate
    // -------------------------------------------------------------------------


    private fun estimateTokensFromChars(chars: Int): Int =
        if (chars <= 0) 0 else (chars + 3) / 4

    private fun extractDemandCategoryFromDeveloperPrompt(developerPrompt: String): String {
        val m = Regex("""demand_category\s*=\s*([A-Z_]+)""")
            .find(developerPrompt)
        return m?.groupValues?.getOrNull(1) ?: "LEGACY_OR_UNSPECIFIED"
    }

    private fun buildChatCompletionsPayload(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        temperature: Double,
        forceJsonObject: Boolean = false
    ): JSONObject {
        val messages = JSONArray().apply {
            put(JSONObject().apply { put("role", "system"); put("content", systemPrompt) })
            put(JSONObject().apply { put("role", "developer"); put("content", developerPrompt) })
            put(JSONObject().apply { put("role", "user"); put("content", userMessage) })
        }

        return JSONObject().apply {
            put("model", model)
            if (forceJsonObject) {
                put("response_format", JSONObject().apply { put("type", "json_object") })
            }
            put("messages", messages)
            put("temperature", temperature)
        }
    }

    private fun extractAssistantContentOrEmpty(bodyString: String): String {
        return try {
            val root = JSONObject(bodyString)
            val choices = root.optJSONArray("choices") ?: JSONArray()
            if (choices.length() == 0) return ""
            choices
                .optJSONObject(0)
                ?.optJSONObject("message")
                ?.optString("content", "")
                ?.trim()
                ?: ""
        } catch (_: Throwable) {
            ""
        }
    }

    private fun intentEnvelopeFallbackEmpty(rawUserText: String?): IntentEnvelopeV1 {
        return IntentEnvelopeV1(
            intents = emptyList(),
            rawUserText = rawUserText,
            language = null,
            asrQuality = null,
            freeTalkTopic = null,
            freeTalkConfidence = 0.0
        )
    }

    // -------------------------------------------------------------------------
    // NEW (Frozen v1): Tick 1 Intent Envelope (NLU only) — no meaning_v1
    // -------------------------------------------------------------------------

    override suspend fun sendIntentEnvelope(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx?
    ): IntentEnvelopeV1 = withContext(Dispatchers.IO) {

        val reqId = UUID.randomUUID().toString().substring(0, 8)
        val t0 = SystemClock.elapsedRealtime()

        var payloadSha12: String? = null
        var httpCode: Int? = null
        var responseBodyForTelemetry: String = ""

        try {
            val payload = buildChatCompletionsPayload(
                systemPrompt = systemPrompt,
                developerPrompt = developerPrompt,
                userMessage = userMessage,
                temperature = 0.0,
                forceJsonObject = true
            )

            val payloadStr = payload.toString()
            payloadSha12 = runCatching { sha256HexUtf8(payloadStr).take(12) }.getOrNull()

            emitModelPayloadOut(
                ctx = telemetryCtx,
                channel = "INTENT",
                reqId = reqId,
                payloadSha12 = payloadSha12,
                payloadStr = payloadStr,
                systemPrompt = systemPrompt,
                developerPrompt = developerPrompt,
                userMessage = userMessage
            )

            val demandCategory =
                telemetryCtx?.demandCategory ?: extractDemandCategoryFromDeveloperPrompt(developerPrompt)

            val systemChars = systemPrompt.length
            val developerChars = developerPrompt.length
            val userChars = userMessage.length
            val staticChars = systemChars + developerChars
            val dynamicChars = userChars
            val outerChars = payloadStr.length

            val systemTokens = estimateTokensFromChars(systemChars)
            val developerTokens = estimateTokensFromChars(developerChars)
            val userTokens = estimateTokensFromChars(userChars)
            val staticTokens = estimateTokensFromChars(staticChars)
            val dynamicTokens = estimateTokensFromChars(dynamicChars)
            val outerTokens = estimateTokensFromChars(outerChars)

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "REPLY_PAYLOAD_METRICS",
                    "turn_id" to telemetryCtx?.turnId,
                    "tick_id" to telemetryCtx?.tickId,
                    "policy_req_seq" to telemetryCtx?.policyReqSeq,
                    "correlation_id" to telemetryCtx?.correlationId,
                    "model_call_id" to telemetryCtx?.modelCallId,
                    "toolplan_id" to telemetryCtx?.toolplanId,
                    "req_id" to reqId,
                    "demand_category" to demandCategory,
                    "rollout_mode" to telemetryCtx?.rolloutMode,
                    "system_chars" to systemChars,
                    "developer_chars" to developerChars,
                    "user_chars" to userChars,
                    "static_chars" to staticChars,
                    "dynamic_chars" to dynamicChars,
                    "outer_payload_chars" to outerChars,
                    "system_tokens_est" to systemTokens,
                    "developer_tokens_est" to developerTokens,
                    "user_tokens_est" to userTokens,
                    "static_tokens_est" to staticTokens,
                    "dynamic_tokens_est" to dynamicTokens,
                    "outer_payload_tokens_est" to outerTokens
                )
            )

            Log.i(
                "SudokuLLM",
                "REPLY payload_metrics req_id=$reqId demand=$demandCategory rollout=${telemetryCtx?.rolloutMode} " +
                        "outer_chars=$outerChars outer_tok_est=$outerTokens static_chars=$staticChars dynamic_chars=$dynamicChars"
            )

            val request = Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(payloadStr.toRequestBody(jsonMediaType))
                .build()

            val httpRes = executeWithRetryPolicy(
                request = request,
                reqId = reqId,
                tag = "INTENT",
                payloadSha12 = payloadSha12 ?: "sha12_missing"
            )

            val dtMs = SystemClock.elapsedRealtime() - t0
            httpCode = httpRes.code
            responseBodyForTelemetry = httpRes.body

            if (!httpRes.ok) {
                emitModelPayloadIn(
                    ctx = telemetryCtx,
                    channel = "INTENT",
                    reqId = reqId,
                    payloadSha12 = payloadSha12,
                    httpCode = httpCode,
                    dtMs = dtMs,
                    responseStr = responseBodyForTelemetry,
                    parseOk = false,
                    parseErrors = "http_${httpRes.code}"
                )

                Log.e("SudokuLLM", "INTENT HTTP ${httpRes.code} req_id=$reqId payload_sha12=$payloadSha12 dtMs=$dtMs body=${capPreview(httpRes.body, 260)}")
                return@withContext intentEnvelopeFallbackEmpty(rawUserText = userMessage)
            }

            val content = extractAssistantContentOrEmpty(httpRes.body)
            if (content.isBlank()) {
                emitModelPayloadIn(
                    ctx = telemetryCtx,
                    channel = "INTENT",
                    reqId = reqId,
                    payloadSha12 = payloadSha12,
                    httpCode = httpCode,
                    dtMs = dtMs,
                    responseStr = responseBodyForTelemetry,
                    parseOk = false,
                    parseErrors = "empty_assistant_content"
                )

                Log.e("SudokuLLM", "INTENT empty content req_id=$reqId dtMs=$dtMs")
                return@withContext intentEnvelopeFallbackEmpty(rawUserText = userMessage)
            }

            val parsed = IntentEnvelopeV1.parseJson(content)

            emitModelPayloadIn(
                ctx = telemetryCtx,
                channel = "INTENT",
                reqId = reqId,
                payloadSha12 = payloadSha12,
                httpCode = httpCode,
                dtMs = dtMs,
                responseStr = responseBodyForTelemetry,
                parseOk = parsed.errors.isEmpty(),
                parseErrors = if (parsed.errors.isEmpty()) null else parsed.errors.joinToString("; ")
            )

            if (parsed.errors.isNotEmpty()) {
                Log.w(
                    "SudokuLLM",
                    "INTENT parsed with errors req_id=$reqId dtMs=$dtMs errors=${parsed.errors.joinToString("; ")} preview=${capPreview(content, 240)}"
                )
            } else {
                Log.i(
                    "SudokuLLM",
                    "INTENT ok req_id=$reqId dtMs=$dtMs intents=${parsed.value.intents.size} freeTalk=${parsed.value.freeTalkTopic ?: "none"}"
                )
            }

            return@withContext parsed.value

        } catch (e: Exception) {
            val dtMs = SystemClock.elapsedRealtime() - t0

            emitModelPayloadIn(
                ctx = telemetryCtx,
                channel = "INTENT",
                reqId = reqId,
                payloadSha12 = payloadSha12,
                httpCode = httpCode,
                dtMs = dtMs,
                responseStr = responseBodyForTelemetry,
                parseOk = false,
                parseErrors = "exception_${e.javaClass.simpleName ?: "Exception"}"
            )

            Log.e("SudokuLLM", "INTENT exception req_id=$reqId dtMs=$dtMs", e)
            return@withContext intentEnvelopeFallbackEmpty(rawUserText = userMessage)
        }
    }

    // -------------------------------------------------------------------------
    // Tick 2 Reply Generate — unchanged behavior, strict {"text":"..."}
    // -------------------------------------------------------------------------

    override suspend fun sendReplyGenerate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx?
    ): String = withContext(Dispatchers.IO) {

        val reqId = UUID.randomUUID().toString().substring(0, 8)
        val t0 = SystemClock.elapsedRealtime()

        var payloadSha12: String? = null
        var httpCode: Int? = null
        var responseBodyForTelemetry: String = ""

        try {
            val payload = buildChatCompletionsPayload(
                systemPrompt = systemPrompt,
                developerPrompt = developerPrompt,
                userMessage = userMessage,
                temperature = 0.2,
                forceJsonObject = true
            )

            val payloadStr = payload.toString()
            payloadSha12 = runCatching { sha256HexUtf8(payloadStr).take(12) }.getOrNull()

            emitModelPayloadOut(
                ctx = telemetryCtx,
                channel = "REPLY",
                reqId = reqId,
                payloadSha12 = payloadSha12,
                payloadStr = payloadStr,
                systemPrompt = systemPrompt,
                developerPrompt = developerPrompt,
                userMessage = userMessage
            )

            val request = Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(payloadStr.toRequestBody(jsonMediaType))
                .build()

            val httpRes = executeWithRetryPolicy(
                request = request,
                reqId = reqId,
                tag = "REPLY",
                payloadSha12 = payloadSha12 ?: "sha12_missing"
            )

            val dtMs = SystemClock.elapsedRealtime() - t0
            httpCode = httpRes.code
            responseBodyForTelemetry = httpRes.body

            if (!httpRes.ok) {
                emitModelPayloadIn(
                    ctx = telemetryCtx,
                    channel = "REPLY",
                    reqId = reqId,
                    payloadSha12 = payloadSha12,
                    httpCode = httpCode,
                    dtMs = dtMs,
                    responseStr = responseBodyForTelemetry,
                    parseOk = false,
                    parseErrors = "http_${httpRes.code}"
                )

                Log.e("SudokuLLM", "REPLY HTTP ${httpRes.code} req_id=$reqId payload_sha12=$payloadSha12 dtMs=$dtMs body=${capPreview(httpRes.body, 260)}")
                return@withContext "Sorry — I didn’t get that. Could you say it once more?"
            }

            val content = extractAssistantContentOrEmpty(httpRes.body)
            if (content.isBlank()) {
                emitModelPayloadIn(
                    ctx = telemetryCtx,
                    channel = "REPLY",
                    reqId = reqId,
                    payloadSha12 = payloadSha12,
                    httpCode = httpCode,
                    dtMs = dtMs,
                    responseStr = responseBodyForTelemetry,
                    parseOk = false,
                    parseErrors = "empty_assistant_content"
                )

                Log.e("SudokuLLM", "REPLY empty content req_id=$reqId dtMs=$dtMs")
                return@withContext "Sorry — I didn’t catch that. Could you repeat?"
            }

            val textOut = runCatching {
                val o = JSONObject(content)
                o.optString("text", "").trim()
            }.getOrDefault("")

            if (textOut.isBlank()) {
                emitModelPayloadIn(
                    ctx = telemetryCtx,
                    channel = "REPLY",
                    reqId = reqId,
                    payloadSha12 = payloadSha12,
                    httpCode = httpCode,
                    dtMs = dtMs,
                    responseStr = responseBodyForTelemetry,
                    parseOk = false,
                    parseErrors = "reply_text_blank_or_json_parse_failed"
                )

                Log.w("SudokuLLM", "REPLY invalid JSON(text) req_id=$reqId dtMs=$dtMs preview=${capPreview(content, 240)}")
                return@withContext "Sorry — I couldn’t render that safely. Please repeat your last message."
            }

            emitModelPayloadIn(
                ctx = telemetryCtx,
                channel = "REPLY",
                reqId = reqId,
                payloadSha12 = payloadSha12,
                httpCode = httpCode,
                dtMs = dtMs,
                responseStr = responseBodyForTelemetry,
                parseOk = true,
                parseErrors = null
            )

            Log.i("SudokuLLM", "REPLY ok req_id=$reqId dtMs=$dtMs len=${textOut.length}")
            return@withContext textOut

        } catch (e: Exception) {
            val dtMs = SystemClock.elapsedRealtime() - t0

            emitModelPayloadIn(
                ctx = telemetryCtx,
                channel = "REPLY",
                reqId = reqId,
                payloadSha12 = payloadSha12,
                httpCode = httpCode,
                dtMs = dtMs,
                responseStr = responseBodyForTelemetry,
                parseOk = false,
                parseErrors = "exception_${e.javaClass.simpleName ?: "Exception"}"
            )

            Log.e("SudokuLLM", "REPLY exception req_id=$reqId dtMs=$dtMs", e)
            return@withContext "Sorry — could you repeat that?"
        }
    }

    // -------------------------------------------------------------------------
    // FREE-TALK MODE — unchanged (telemetry/ID silent)
    // -------------------------------------------------------------------------

    override suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): FreeTalkRawResponse {
        return chatFreeTalk(systemPrompt, developerPrompt, userMessage, telemetryCtx = null)
    }

    suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx?
    ): FreeTalkRawResponse = withContext(Dispatchers.IO) {
        val reqId = UUID.randomUUID().toString().substring(0, 8)
        val t0 = SystemClock.elapsedRealtime()

        try {
            fun capForHistory(role: String, text: String): String {
                val maxChars = when (role) {
                    "user" -> 900
                    "assistant" -> 650
                    else -> 650
                }
                val normalized = text.replace("\r\n", "\n").trimEnd()
                if (normalized.length <= maxChars) return normalized
                return normalized.substring(0, maxChars).trimEnd() + "…"
            }

            data class ParsedHistory(
                val ok: Boolean,
                val pre: String,
                val history: String,
                val post: String,
                val items: List<Pair<String, String>>,
                val userCount: Int,
                val assistantCount: Int
            )

            fun extractBlockOrNull(text: String, begin: String, end: String): String? {
                val bi = text.indexOf(begin)
                val ei = text.indexOf(end)
                if (bi < 0 || ei < 0 || ei <= bi) return null
                return text.substring(bi + begin.length, ei).trim()
            }

            fun capChars(s: String, maxChars: Int): String {
                val normalized = s.replace("\r\n", "\n").trimEnd()
                if (normalized.length <= maxChars) return normalized
                return normalized.substring(0, maxChars).trimEnd() + "…"
            }

            fun buildPinnedGridContextSystemMessage(developerPrompt: String): String {
                val devTrim = developerPrompt.trim()
                val marked = extractBlockOrNull(devTrim, "BEGIN_GRID_CONTEXT", "END_GRID_CONTEXT")
                val payload = (marked ?: devTrim).trim()
                val capped = capChars(payload, 8000)

                return buildString {
                    appendLine("GRID CONTEXT (AUTHORITATIVE):")
                    appendLine("- The following block describes the CURRENT captured grid from the user's scan.")
                    appendLine("- Treat it as ground truth and do not contradict it.")
                    appendLine("- If you need a detail not present here, ask the user to confirm a specific cell.")
                    appendLine()
                    appendLine(capped)
                }.trim()
            }

            fun parseCanonicalHistory(dev: String): ParsedHistory {
                val begin = "BEGIN_CANONICAL_HISTORY"
                val end = "END_CANONICAL_HISTORY"
                val bi = dev.indexOf(begin)
                val ei = dev.indexOf(end)

                if (bi < 0 || ei < 0 || ei <= bi) {
                    return ParsedHistory(false, dev, "", "", emptyList(), 0, 0)
                }

                val pre = dev.substring(0, bi).trim()
                val history = dev.substring(bi + begin.length, ei).trim()
                val post = dev.substring(ei + end.length).trim()

                val items = ArrayList<Pair<String, String>>()
                val lines = history.split('\n')

                var curRole: String? = null
                val cur = StringBuilder()

                fun flush() {
                    val r = curRole
                    if (r == "user" || r == "assistant") {
                        val txt = cur.toString().trim()
                        if (txt.isNotEmpty()) items.add(r to txt)
                    }
                    curRole = null
                    cur.setLength(0)
                }

                for (raw in lines) {
                    val line = raw.trimEnd()
                    val m = Regex("^(system|developer|user|assistant)\\s*:\\s?(.*)$", RegexOption.IGNORE_CASE).find(line)

                    if (m != null) {
                        flush()
                        val roleRaw = m.groupValues[1].lowercase()
                        val contentStart = m.groupValues[2]
                        curRole = roleRaw
                        cur.append(contentStart)
                        cur.append('\n')
                    } else {
                        if (curRole != null) {
                            cur.append(line)
                            cur.append('\n')
                        }
                    }
                }
                flush()

                val u = items.count { it.first == "user" }
                val a = items.count { it.first == "assistant" }
                return ParsedHistory(true, pre, history, post, items, u, a)
            }

            fun budgetHistory(items: List<Pair<String, String>>, maxMsgs: Int = 14): List<Pair<String, String>> {
                if (items.isEmpty()) return emptyList()
                val tail = if (items.size > maxMsgs) items.takeLast(maxMsgs) else items
                return tail.map { (role, content) -> role to capForHistory(role, content) }
            }

            val sysTrim = augmentSystemPrompt(systemPrompt)
            val devTrim = developerPrompt.trim()
            val usrTrim = userMessage.trim()

            val parsed = parseCanonicalHistory(devTrim)
            val budgeted = budgetHistory(parsed.items)
            val pinnedGrid = buildPinnedGridContextSystemMessage(devTrim)

            val contextBlob = buildString {
                val pre = parsed.pre.trim().takeIf { it.isNotEmpty() }?.let { capChars(it, 5000) }
                val post = parsed.post.trim().takeIf { it.isNotEmpty() }?.let { capChars(it, 5000) }

                appendLine("BEGIN_CONTEXT")
                if (pre != null) { appendLine(pre); appendLine() }
                if (post != null) { appendLine(post); appendLine() }

                appendLine("TURN-TAKING / MEMORY CONTRACT (STRICT):")
                appendLine("- The conversation history is provided as actual user/assistant messages below.")
                appendLine("- If the user asks you to repeat what you said earlier, quote the most recent assistant message verbatim from provided history.")
                appendLine("- Do not claim you cannot recall if the answer is present in the provided history.")
                appendLine("END_CONTEXT")
            }.trim()

            val messages = JSONArray().apply {
                put(JSONObject().apply { put("role", "system"); put("content", sysTrim) })
                put(JSONObject().apply { put("role", "system"); put("content", pinnedGrid) })
                if (contextBlob.isNotBlank()) put(JSONObject().apply { put("role", "system"); put("content", contextBlob) })
                for ((role, content) in budgeted) put(JSONObject().apply { put("role", role); put("content", content) })
                put(JSONObject().apply { put("role", "user"); put("content", usrTrim) })
            }

            val payload = JSONObject().apply {
                put("model", model)
                put("messages", messages)
                put("temperature", 0.7)
            }

            val payloadStr = payload.toString()
            val payloadSha12 = sha256HexUtf8(payloadStr).take(12)

            val request = Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(payloadStr.toRequestBody(jsonMediaType))
                .build()

            val response = httpClient.newCall(request).execute()
            val bodyString = response.body?.string() ?: ""
            val dtMs = SystemClock.elapsedRealtime() - t0

            if (!response.isSuccessful) {
                Log.e("SudokuLLM", "FREE_TALK HTTP ${response.code} req_id=$reqId payload_sha12=$payloadSha12 body=${capPreview(bodyString, 260)}")
                return@withContext FreeTalkRawResponse("I’m having trouble reaching the server. Want to try again?")
            }

            val root = JSONObject(bodyString)
            val usedModel = root.optString("model", "<missing>")
            Log.i("SudokuLLM", "OpenAI FREE_TALK response model=$usedModel (requested=$model) req_id=$reqId dtMs=$dtMs")

            val content = root.getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
                .trim()

            return@withContext FreeTalkRawResponse(content)

        } catch (e: Exception) {
            val dtMs = SystemClock.elapsedRealtime() - t0
            Log.e("SudokuLLM", "FREE_TALK exception req_id=$reqId dtMs=$dtMs", e)
            return@withContext FreeTalkRawResponse("Something went wrong on my side. Can you repeat that?")
        }
    }

    // -------------------------------------------------------------------------
    // extractClues — unchanged behavior (still telemetry/ID silent)
    // -------------------------------------------------------------------------

    override suspend fun extractClues(
        systemPrompt: String,
        developerPrompt: String,
        transcript: String
    ): ClueExtractionRawResponse = withContext(Dispatchers.IO) {
        try {
            val mergedUserContent = buildString {
                appendLine(developerPrompt.trim())
                appendLine()
                appendLine("Transcript to analyze:")
                appendLine(transcript.trim())
                appendLine()
                appendLine("""Return ONLY JSON: {"clues":[{"key":"...","value":"...","confidence":0.0}]}""")
            }

            val messages = JSONArray().apply {
                put(JSONObject().apply { put("role", "system"); put("content", augmentSystemPrompt(systemPrompt)) })
                put(JSONObject().apply { put("role", "user"); put("content", mergedUserContent) })
            }

            val payload = JSONObject().apply {
                put("model", model)
                put("response_format", JSONObject().apply { put("type", "json_object") })
                put("messages", messages)
                put("temperature", 0.2)
            }

            val request = Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(payload.toString().toRequestBody(jsonMediaType))
                .build()

            val response = httpClient.newCall(request).execute()
            val bodyString = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                Log.e("SudokuLLM", "OpenAI extractClues error: HTTP ${response.code} - ${capPreview(bodyString, 260)}")
                return@withContext ClueExtractionRawResponse(emptyList())
            }

            val content = JSONObject(bodyString)
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
                .trim()

            val obj = JSONObject(content)
            val arr = obj.optJSONArray("clues") ?: JSONArray()

            val clues = List(arr.length()) { i ->
                val c = arr.getJSONObject(i)
                UserClue(
                    key = c.optString("key", "").trim(),
                    value = c.optString("value", "").trim(),
                    confidence = (c.optDouble("confidence", 0.7)).toFloat(),
                    source = "conversation"
                )
            }.filter { it.key.isNotBlank() && it.value.isNotBlank() }

            ClueExtractionRawResponse(clues)

        } catch (e: Exception) {
            Log.e("SudokuLLM", "Error in extractClues()", e)
            ClueExtractionRawResponse(emptyList())
        }
    }
}