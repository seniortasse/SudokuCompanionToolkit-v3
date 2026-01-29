package com.contextionary.sudoku.logic

import android.os.SystemClock
import android.util.Log
import com.contextionary.sudoku.conductor.SudoToolJsonSchema
import com.contextionary.sudoku.telemetry.ConversationTelemetry
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

/**
 * Real LLM client using OpenAI's chat completions endpoint.
 *
 * GRID MODE:
 * - Uses Structured Outputs (json_schema strict) to force a tool plan every time.
 * - Parses tool_calls only (no free-form assistant_message).
 * - If parsing fails or tool_calls missing, falls back to a mission-aligned repair plan.
 *
 * FREE-TALK MODE:
 * - Kept as-is (uses regular text response).
 */
class RealSudokuLLMClient(
    private val apiKey: String,
    private val model: String
) : SudokuLLMClient {

    private val httpClient = OkHttpClient()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    // -------------------------------------------------------------------------
    // Step 2 — Expert invariants + glossary + grounding contract (pinned).
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

Case 1 — mismatch_indices_vs_deduced is not empty:
- Pick ONE cell from mismatch_indices_vs_deduced (one at a time).
- Explain: printed clues force a different value here.
- You MAY propose the specific digit ONLY if GRID_CONTEXT provides it.
- Ask the user to confirm what’s on paper using ask_confirm_cell_rc.

Case 2 — solvability == "none" AND mismatch list is empty:
- Pick ONE cell from unresolved_indices (one at a time).
- Justify why you picked it (conflict explanation if available, otherwise uncertainty).
- Do NOT invent a replacement digit unless GRID_CONTEXT provides a target digit.
- Ask the user what the correct digit is (or whether it should be blank).

Case 3 — solvability == "unique" AND mismatch list is empty:
- STOP corrections. Do NOT propose cell checks.
- Ask if the on-screen grid matches paper exactly (recommend_validate).

Case 4 — solvability == "multiple" AND mismatch list is empty:
- Do NOT do cell-by-cell loops.
- Ask if on-screen grid is a 100% match (recommend_validate).
- If user says YES and it remains multiple -> recommend_retake.
- If user says NO -> ask for one specific differing cell; apply_user_edit_rc on request.

H) ONE-STEP DISCIPLINE
- Only propose ONE actionable step per turn.
- If you propose a single cell check, you MUST also ask that cell using ask_confirm_cell_rc(row,col,...).
- Never propose a cell outside mismatch_indices_vs_deduced or unresolved_indices (and unresolved only when solvability=="none").

I) Tool usage reminder (GRID MODE)
- Every response MUST include reply(text="...") with non-empty text.
- Use ask_confirm_cell_rc for targeted verification.
- Use recommend_validate for “does it match / are we good to proceed?” (especially Case 3 and Case 4).
- Use recommend_retake when the state cannot be made solve-ready without a better scan.


TECHNICAL JARGON (APP INTERNAL — DO NOT SAY THESE WORDS TO THE USER)
You will receive GRID_CONTEXT blocks from the app. They contain internal vocabulary.
You must understand these definitions to interpret the grid correctly, but you must not use these labels in your conversation. Instead, translate to normal Sudoku language (e.g., “given,” “filled-in number,” “pencil marks,” “row 1 column 2,” “there’s a contradiction,” “please confirm this cell,” etc.).
0) Core objects
•	cell / square: One position in the 9×9 grid.
•	digit: A value in a cell. Allowed digits are 1–9.
•	blank: A cell whose digit is 0 (meaning empty / not filled).
1) Coordinates & indexing
•	row / r1..r9: Row number 1 to 9.
•	col / c1..c9: Column number 1 to 9.
•	rXcY: Cell coordinate (X,Y in 1..9). Example: r4c7 = row 4, column 7.
•	index / idx (0..80): Zero-based cell index:
o	idx = (row-1)*9 + (col-1)
o	row = idx/9 + 1, col = idx%9 + 1
2) Digits and blanks
•	digit: A value 0..9 where:
o	0 means blank (empty cell)
o	1..9 are normal Sudoku digits
When you describe a row/column to the user, use row/col language, not “idx”.
3) The 3 “heads” from the CellInterpreter (IMPORTANT)
The vision model predicts three independent outputs (“heads”) per cell:
A) GIVEN head (printed givens)
•	Meaning: What digit is printed in the puzzle’s original grid (the puzzle’s immutable starting clues).
•	In the app: This is part of the puzzle DNA.
•	In GRID_CONTEXT: appears as TRUTH_LAYER_GIVENS (FACTS / DNA).
B) SOLUTION head (placed answers)
•	Meaning: A digit that is an answer placed in the cell (not a tiny pencil mark).
•	At grid capture: usually a handwritten user answer detected from the scanned source.
•	After capture: the app may also contain solver-provided placed answers (they are still “placed answers” even if not handwritten).
•	Important: In the current pipeline, this head is “solution/answer head” — not strictly “handwritten” forever.
•	In GRID_CONTEXT: appears as TRUTH_LAYER_USER_SOLUTIONS (USER CLAIMS / OPINIONS) when the origin is the user.
o	Treat it as a claim unless confirmed.
o	If later the app marks solver origin separately, treat solver-origin placed answers as higher-trust than user-origin answers.
C) CANDIDATES head (pencil marks)
•	Meaning: Tiny candidate digits the user penciled in as possibilities, not final answers.
•	Output form: a candidateMask (bitmask for digits 1..9) derived from sigmoid probabilities with a threshold.
•	This is a separate head from solutions and givens.
•	In GRID_CONTEXT: appears under CANDIDATES (USER THOUGHT PROCESS; MAY BE NOISY).
4) “Truth layers” vs “Current display”
The app keeps multiple representations:
•	CURRENT_DISPLAY_DIGITS_0_MEANS_BLANK
o	What is currently shown on screen in the grid.
o	This is the authoritative answer to “what do you see in row 1?” because it’s what the user is looking at.
•	TRUTH_LAYER_GIVENS (FACTS / DNA)
o	The model’s belief of which digits are printed givens.
o	Treated as “facts” for reasoning, but the user can override if scan error.
•	TRUTH_LAYER_USER_SOLUTIONS (USER CLAIMS / OPINIONS)
o	The model’s belief of which digits are user-placed answers (usually handwritten at capture).
o	Treat as “user claims”, may be wrong.
•	CANDIDATES
o	Candidate marks (pencil marks), noisy and incomplete.
5) Conflict vs unresolved (do NOT confuse these)
conflict (conflict_indices)
A hard Sudoku rule violation in the current display:
•	The same non-zero digit appears twice in a row, column, or 3×3 box.
•	Conflicts imply the grid is structurally invalid.
User-facing translation:
•	Say: “I’m seeing a contradiction: there are two 7s in this row/column/box.”
unresolved (unresolved_indices)
An app-specific “needs verification” set used when the app did not reach a clean trustworthy state.
In the current auto-corrector logic, unresolved typically includes:
•	cells that are still in conflict, and/or
•	low-confidence cells that were not changed by auto-correction,
•	especially when the final grid is not a consistent unique-solution state.
User-facing translation:
•	Say: “A few cells look uncertain — can you quickly confirm them?”
6) Auto-correction & edit provenance
•	auto_changed_indices
o	Cells where auto-correction changed the digit (app made a guess).
o	User-facing translation: “I adjusted a couple of cells as a best guess—let’s verify them.”
•	manual_corrected_indices / manualEdits
o	Cells the user explicitly edited/confirmed inside the app.
o	Treat as higher-trust than auto-changes.
7) Solver facts from givens only (separate from “current display solvability”)
•	deduced_solution_count_capped (0/1/2)
o	0 = no solution from givens only
o	1 = unique solution from givens only
o	2 = multiple solutions (2+), capped at 2
•	deduced_is_unique
o	True when givens-only solution is unique and a full solution grid is provided.
•	DEDUCED_UNIQUE_SOLUTION_GRID
o	Present only when deduced_is_unique = true.
•	mismatch_indices_vs_deduced / mismatchDetails
o	Only relevant when givens-only solution is unique.
o	Marks where user-placed answers conflict with the deduced solution.
User-facing translation:
•	“One of the numbers you entered doesn’t match what the printed clues force. Let’s verify that cell.”
8) Status fields
•	solvability_of_current_display: unique | multiple | none
o	Applies to the current display grid.
•	is_structurally_valid
o	True if there are no conflicts in current display.
•	severity: ok | mild | serious
o	App seriousness score used to guide whether to verify vs retake.
•	retake_recommendation: none | soft | strong
o	Internal suggestion about rescanning quality.
9) Rule: never expose jargon
In conversation:
•	Do NOT say: “truth layer”, “DNA”, “idx”, “candidateMask”, “deduced_solution_count_capped”, “structurally valid”, “severity=serious”, etc.
•	DO say: “printed clues”, “numbers you already filled”, “pencil marks”, “contradiction”, “can you confirm row X column Y”.
""${'"'}.
""".trimIndent()



    private fun augmentSystemPrompt(base: String): String {
        val b = base.trim()

        val hardOverride = """
        HARD OVERRIDE (takes precedence over any earlier instructions)
        
        
        0) MANDATORY PENDING RESOLUTION (HARD — NO EXCEPTIONS)

        If STATE_HEADER / pending_ctx indicates ask_cell_value with a specific (row,col),
        and the user message contains a clear digit (1..9) or clearly says blank/empty:

        - You MUST emit confirm_cell_value_rc(row,col,digit_or_blank) in this same response.
        - If the confirmed value differs from CURRENT_DISPLAY at that cell, you MUST ALSO emit apply_user_edit_rc(row,col,digit_or_blank,source=...).
        - You MUST NOT proceed to the next check (ask_confirm_cell_rc) unless confirm_cell_value_rc has been emitted for the pending cell first.
        - If the digit is not clear, you MUST emit ask_clarifying_question (do not “guess”).

        This rule overrides any softer instruction about “only apply edits when explicitly requested” for the pending cell.
        
        
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
           Case 1 — mismatch_indices_vs_deduced is not empty:
             - Pick ONE mismatch cell, explain “printed clues force a different number here.”
             - Ask that cell with ask_confirm_cell_rc.
             - Only propose a replacement digit if GRID_CONTEXT explicitly provides it.

           Case 2 — solvability == "none" AND mismatch empty:
             - Pick ONE cell from unresolved_indices.
             - If it is also a conflict cell, explain using CONFLICTS_DETAILS.
             - Otherwise say it needs quick verification.
             - Ask that cell with ask_confirm_cell_rc.
             - Do NOT invent a replacement digit.

           Case 3 — solvability == "unique" AND mismatch empty:
             - STOP all correction/check behavior immediately.
             - Do NOT ask for unresolved/conflict checks.
             - Offer the grid “as-is” and ask whether the on-screen grid matches the paper (recommend_validate).

           Case 4 — solvability == "multiple" AND mismatch empty:
             - Do NOT run cell-by-cell verification.
             - Tell the user the grid admits multiple solutions and Sudo can only assist on unique grids.
             - Ask: “Is the on-screen grid a 100% match with your paper?”
             - If user says YES and solvability remains multiple -> recommend a retake.
             - If user says NO -> the user will identify a specific differing cell; apply_user_edit_rc on request.

        4) CONFLICT GROUNDING (STRICT)
           - You may ONLY claim a contradiction if it appears in CONFLICTS_DETAILS.
           - If CONFLICTS_DETAILS says “(none)”, you MUST NOT claim any contradiction.
    """.trimIndent()

        return buildString {
            appendLine(b)
            appendLine()
            appendLine("-----")
            appendLine(SUDOKU_EXPERT_BLOCK)
            appendLine()
            appendLine("-----")
            appendLine(hardOverride)
        }.trim()
    }

    // -------------------------------------------------------------------------
    // Step 1 — GridContext pinning helpers
    // -------------------------------------------------------------------------

    private fun extractBlockOrNull(
        text: String,
        begin: String,
        end: String
    ): String? {
        val bi = text.indexOf(begin)
        val ei = text.indexOf(end)
        if (bi < 0 || ei < 0 || ei <= bi) return null
        return text.substring(bi + begin.length, ei).trim()
    }

    private fun buildPinnedGridContextSystemMessage(developerPrompt: String): String {
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

    private fun buildPinnedCaptureOriginSystemMessage(developerPrompt: String): String {
        val devTrim = developerPrompt.trim()
        val origin = extractBlockOrNull(devTrim, "BEGIN_CAPTURE_ORIGIN", "END_CAPTURE_ORIGIN")
        val payload = (origin ?: "User scanned a real Sudoku grid using the phone camera.").trim()
        return buildString {
            appendLine("CAPTURE ORIGIN (CONTEXT):")
            appendLine(payload)
        }.trim()
    }

    private fun capChars(s: String, maxChars: Int): String {
        val normalized = s.replace("\r\n", "\n").trimEnd()
        if (normalized.length <= maxChars) return normalized
        return normalized.substring(0, maxChars).trimEnd() + "…"
    }

    // -------------------------------------------------------------------------
    // JSON helpers
    // -------------------------------------------------------------------------

    private fun org.json.JSONObject.toKotlinMap(): Map<String, Any?> {
        val out = linkedMapOf<String, Any?>()
        val it = keys()
        while (it.hasNext()) {
            val k = it.next()
            val v = opt(k)
            out[k] = when (v) {
                is org.json.JSONObject -> v.toKotlinMap()
                is org.json.JSONArray -> v.toKotlinList()
                org.json.JSONObject.NULL -> null
                else -> v
            }
        }
        return out
    }

    private fun org.json.JSONArray.toKotlinList(): List<Any?> {
        val out = ArrayList<Any?>(length())
        for (i in 0 until length()) {
            val v = opt(i)
            out += when (v) {
                is org.json.JSONObject -> v.toKotlinMap()
                is org.json.JSONArray -> v.toKotlinList()
                org.json.JSONObject.NULL -> null
                else -> v
            }
        }
        return out
    }

    // -------------------------------------------------------------------------
    // GRID MODE (tool-calls-only)
    // -------------------------------------------------------------------------

    override suspend fun sendGridUpdate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        history: List<Pair<String, String>>
    ): PolicyRawResponse = withContext(Dispatchers.IO) {

        val reqId = UUID.randomUUID().toString().substring(0, 8)
        val t0 = SystemClock.elapsedRealtime()

        val enforceToolPlanOk = false
        val DEBUG_DUMP_LLM_PROMPT = true

        var toolplanFinalized = false

        fun sha256HexUtf8(s: String): String {
            val md = MessageDigest.getInstance("SHA-256")
            val digest = md.digest(s.toByteArray(Charsets.UTF_8))
            val hex = StringBuilder(digest.size * 2)
            for (b in digest) hex.append(String.format("%02x", b))
            return hex.toString()
        }

        fun capPreview(text: String, maxChars: Int): String {
            val normalized = text
                .replace('\n', ' ')
                .replace('\r', ' ')
                .replace(Regex("\\s+"), " ")
                .trim()
            if (normalized.length <= maxChars) return normalized
            return normalized.substring(0, maxChars).trimEnd() + "…"
        }

        fun emitLargeText(eventType: String, label: String, text: String, chunkSize: Int = 1800) {
            val normalized = text.replace("\r\n", "\n")
            val total = normalized.length
            val chunks = if (total == 0) 1 else ((total + chunkSize - 1) / chunkSize)

            for (i in 0 until chunks) {
                val start = i * chunkSize
                val end = minOf(total, start + chunkSize)
                val part = if (total == 0) "" else normalized.substring(start, end)

                ConversationTelemetry.emitKv(
                    eventType,
                    "req_id" to reqId,
                    "mode" to "GRID",
                    "label" to label,
                    "chunk_index" to i,
                    "chunk_count" to chunks,
                    "text_part" to part
                )
            }
        }

        fun emitPromptDump(
            sysAfterAugment: String,
            devTrim: String,
            usrTrim: String,
            messagesPretty: String,
            payloadPretty: String,
            payloadSha256: String
        ) {
            ConversationTelemetry.emitKv(
                "LLM_PROMPT_DUMP_META",
                "req_id" to reqId,
                "mode" to "GRID",
                "model" to model,
                "sys_len" to sysAfterAugment.length,
                "dev_len" to devTrim.length,
                "usr_len" to usrTrim.length,
                "messages_pretty_len" to messagesPretty.length,
                "payload_pretty_len" to payloadPretty.length,
                "payload_sha256" to payloadSha256
            )
            emitLargeText("LLM_PROMPT_DUMP", "sys_after_augment", sysAfterAugment)
            emitLargeText("LLM_PROMPT_DUMP", "devTrim", devTrim)
            emitLargeText("LLM_PROMPT_DUMP", "usrTrim", usrTrim)
            emitLargeText("LLM_PROMPT_DUMP", "messages_pretty_json", messagesPretty)
            emitLargeText("LLM_PROMPT_DUMP", "payload_pretty_json", payloadPretty)
        }

        fun emitToolPlanOkOnce(source: String, tools: List<ToolCallRaw>) {
            if (toolplanFinalized) return
            toolplanFinalized = true
            val names = tools.map { it.name }.joinToString(",")
            ConversationTelemetry.emitKv(
                "LLM_TOOLPLAN_OK",
                "req_id" to reqId,
                "mode" to "GRID",
                "source" to source,
                "tool_count" to tools.size,
                "tool_names" to names
            )
        }

        fun emitToolPlanFallbackOnce(kind: String, extras: Map<String, Any?> = emptyMap()) {
            if (toolplanFinalized) return
            toolplanFinalized = true
            val base = linkedMapOf<String, Any?>(
                "req_id" to reqId,
                "mode" to "GRID",
                "kind" to kind
            )
            for ((k, v) in extras) base[k] = v
            ConversationTelemetry.emitKv(
                "LLM_TOOLPLAN_FALLBACK",
                *base.entries.map { it.key to it.value }.toTypedArray()
            )
        }

        fun replyOnly(text: String): PolicyRawResponse =
            PolicyRawResponse(
                tool_calls = listOf(
                    ToolCallRaw(name = "reply", args = mapOf("text" to text))
                )
            )

        fun fallbackToolsFromUserText(userTextRaw: String): PolicyRawResponse {
            emitToolPlanFallbackOnce("generic_repair")
            return replyOnly(
                "I didn’t catch that clearly. Please tell me ONE cell using: “row 2 column 1 is 5” or “r2 c1 is 5”."
            )
        }

        fun parseToolCallsSafe(contentJson: JSONObject?, userText: String): PolicyRawResponse {
            if (contentJson == null) return fallbackToolsFromUserText(userText)

            val arr = contentJson.optJSONArray("tool_calls") ?: JSONArray()
            val tools = mutableListOf<ToolCallRaw>()

            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                val name = obj.optString("name", "").trim()
                if (name.isEmpty()) continue
                val argsObj = obj.optJSONObject("args") ?: JSONObject()
                tools += ToolCallRaw(name = name, args = argsObj.toKotlinMap())
            }

            if (tools.isEmpty()) {
                emitToolPlanFallbackOnce("empty_tool_calls", mapOf("usr_len" to userText.length))
                return fallbackToolsFromUserText(userText)
            }

            val hasReply = tools.any { it.name.trim().equals("reply", ignoreCase = true) }
            if (!hasReply) {
                emitToolPlanFallbackOnce(
                    "missing_reply_tool",
                    mapOf(
                        "usr_len" to userText.length,
                        "tool_names" to tools.joinToString(",") { it.name }
                    )
                )
                return PolicyRawResponse(tool_calls = emptyList())
            }

            emitToolPlanOkOnce(source = "llm", tools = tools)
            return PolicyRawResponse(tool_calls = tools)
        }

        fun classifyToolPlan(content: String): Pair<String, Int> {
            if (content.isBlank()) return "blank_content" to 0
            val obj = try { JSONObject(content) } catch (_: Exception) { return "not_json" to 0 }
            val arr = obj.optJSONArray("tool_calls") ?: return "missing_tool_calls" to 0
            if (arr.length() == 0) return "empty_tool_calls" to 0
            return "ok" to arr.length()
        }

        // History budget (history is now passed explicitly)
        data class HistMsg(val role: String, val content: String)

        fun budgetHistory(
            msgs: List<HistMsg>,
            maxMessages: Int = 16,
            maxCharsTotal: Int = 9000,
            maxCharsPerMsg: Int = 1400
        ): List<HistMsg> {
            if (msgs.isEmpty()) return emptyList()
            val tail = if (msgs.size > maxMessages) msgs.takeLast(maxMessages) else msgs
            val clipped = tail.map { m ->
                val c = m.content
                    .replace("\r\n", "\n")
                    .trim()
                    .let { if (it.length <= maxCharsPerMsg) it else it.substring(0, maxCharsPerMsg).trimEnd() + "…" }
                HistMsg(m.role, c)
            }
            val out = mutableListOf<HistMsg>()
            var used = 0
            for (m in clipped) {
                val add = m.content.length
                if (used + add > maxCharsTotal) break
                used += add
                out += m
            }
            return out
        }

        try {
            val sys = augmentSystemPrompt(systemPrompt)
            val devTrim = developerPrompt.trim()
            val usrTrim = userMessage.trim()

            // Defensive normalize + drop trailing dup of current user (if any)
            fun norm(s: String) = s.trim().replace(Regex("\\s+"), " ")

            val historyPairs = history
                .mapNotNull { (role, content) ->
                    val r = role.lowercase().trim()
                    if (r != "user" && r != "assistant") null
                    else {
                        val c = content.replace("\r\n", "\n").trimEnd()
                        if (c.isBlank()) null else (r to c)
                    }
                }
                .let { h ->
                    val last = h.lastOrNull()
                    if (last != null && last.first == "user" && norm(last.second) == norm(usrTrim)) h.dropLast(1) else h
                }

            val histBudgeted = budgetHistory(historyPairs.map { (r, c) -> HistMsg(r, c) })

            ConversationTelemetry.emitKv(
                "SEND_GRID_UPDATE_CALLED",
                "req_id" to reqId,
                "mode" to "GRID",
                "user_text" to capPreview(usrTrim, 220),
                "history_msgs_in" to historyPairs.size,
                "history_msgs_used" to histBudgeted.size
            )

            val messages = JSONArray().apply {
                put(JSONObject().apply { put("role", "system"); put("content", sys) })
                put(JSONObject().apply { put("role", "system"); put("content", devTrim) })

                for (m in histBudgeted) {
                    put(JSONObject().apply {
                        put("role", m.role) // "user" or "assistant"
                        put("content", m.content)
                    })
                }

                put(JSONObject().apply {
                    put("role", "user")
                    put("content", if (usrTrim.isBlank()) "Continue." else usrTrim)
                })
            }

            val payload = JSONObject().apply {
                put("model", model)
                put("response_format", SudoToolJsonSchema.responseFormat())
                put("messages", messages)
                put("temperature", 0.4)
            }

            val payloadStr = payload.toString()
            val payloadSha = sha256HexUtf8(payloadStr)

            if (DEBUG_DUMP_LLM_PROMPT) {
                val messagesPretty = try { messages.toString(2) } catch (_: Exception) { messages.toString() }
                val payloadPretty = try { payload.toString(2) } catch (_: Exception) { payload.toString() }
                emitPromptDump(
                    sysAfterAugment = sys,
                    devTrim = devTrim,
                    usrTrim = usrTrim,
                    messagesPretty = messagesPretty,
                    payloadPretty = payloadPretty,
                    payloadSha256 = payloadSha
                )
            }

            ConversationTelemetry.emitKv(
                "LLM_HTTP_BEGIN",
                "req_id" to reqId,
                "mode" to "GRID",
                "model" to model,
                "sys_len" to sys.length,
                "dev_len" to devTrim.length,
                "usr_len" to usrTrim.length,
                "payload_bytes" to payloadStr.toByteArray(Charsets.UTF_8).size,
                "payload_sha256" to payloadSha,
                "response_format" to "json_schema_strict",
                "schema_name" to "sudo_policy",
                "schema_sha256" to SudoToolJsonSchema.schemaSha256(),
                "messages_count" to messages.length(),
                "history_msgs_used" to histBudgeted.size
            )

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
                ConversationTelemetry.emitKv(
                    "LLM_HTTP_END",
                    "req_id" to reqId,
                    "mode" to "GRID",
                    "model" to model,
                    "http_code" to response.code,
                    "ok" to false,
                    "latency_ms" to dtMs,
                    "body_len" to bodyString.length,
                    "body_preview" to capPreview(bodyString, 240)
                )
                emitToolPlanFallbackOnce(
                    "http_error",
                    mapOf("http_code" to response.code, "body_preview" to capPreview(bodyString, 240))
                )
                return@withContext replyOnly(
                    "I couldn’t get a valid tool plan right now (code ${response.code}). " +
                            "Tell me ONE cell like “row 1 column 1 is 5”, or tap a cell and say the digit (1–9)."
                )
            }

            val root = JSONObject(bodyString)
            val usedModel = root.optString("model", "<missing>")
            Log.i("SudokuLLM", "OpenAI GRID response model = $usedModel (requested=$model, req_id=$reqId)")

            val choices = root.optJSONArray("choices") ?: JSONArray()
            if (choices.length() == 0) {
                ConversationTelemetry.emitKv(
                    "LLM_HTTP_END",
                    "req_id" to reqId,
                    "mode" to "GRID",
                    "model" to model,
                    "http_code" to response.code,
                    "ok" to false,
                    "latency_ms" to dtMs,
                    "error" to "no_choices"
                )
                emitToolPlanFallbackOnce("no_choices")
                return@withContext replyOnly(
                    "I didn’t get a valid action back. Tell me ONE cell like “row 1 column 1 is 5”, or tap a cell and say the digit (1–9)."
                )
            }

            val content = choices
                .optJSONObject(0)
                ?.optJSONObject("message")
                ?.optString("content", "")
                ?.trim()
                ?: ""

            val (verdict, toolCount) = classifyToolPlan(content)
            ConversationTelemetry.emitKv(
                "LLM_TOOLPLAN_VERDICT",
                "req_id" to reqId,
                "mode" to "GRID",
                "verdict" to verdict,
                "tool_count" to toolCount,
                "content_len" to content.length,
                "content_sha256" to sha256HexUtf8(content),
                "content_preview" to capPreview(content, 220)
            )

            if (enforceToolPlanOk && verdict != "ok") {
                throw IllegalStateException("LLM tool plan invalid: $verdict :: ${capPreview(content, 260)}")
            }

            val contentJson = try {
                JSONObject(content)
            } catch (_: Exception) {
                ConversationTelemetry.emitKv(
                    "LLM_HTTP_END",
                    "req_id" to reqId,
                    "mode" to "GRID",
                    "model" to model,
                    "http_code" to response.code,
                    "ok" to true,
                    "latency_ms" to dtMs,
                    "parse_ok" to false,
                    "content_len" to content.length
                )
                emitToolPlanFallbackOnce(
                    "content_not_json",
                    mapOf(
                        "content_len" to content.length,
                        "content_sha256" to sha256HexUtf8(content),
                        "content_preview" to capPreview(content, 240)
                    )
                )
                return@withContext fallbackToolsFromUserText(usrTrim)
            }

            ConversationTelemetry.emitKv(
                "LLM_HTTP_END",
                "req_id" to reqId,
                "mode" to "GRID",
                "model" to model,
                "http_code" to response.code,
                "ok" to true,
                "latency_ms" to dtMs,
                "parse_ok" to true
            )

            return@withContext parseToolCallsSafe(contentJson, usrTrim)

        } catch (e: Exception) {
            val dtMs = SystemClock.elapsedRealtime() - t0
            ConversationTelemetry.emitKv(
                "LLM_HTTP_END",
                "req_id" to reqId,
                "mode" to "GRID",
                "model" to model,
                "ok" to false,
                "latency_ms" to dtMs,
                "exception" to (e.javaClass.simpleName ?: "Exception"),
                "message" to (e.message ?: "")
            )

            emitToolPlanFallbackOnce(
                "exception",
                mapOf(
                    "exception" to (e.javaClass.simpleName ?: "Exception"),
                    "message" to (e.message ?: "")
                )
            )

            return@withContext replyOnly(
                "Something went wrong while preparing the tool plan. " +
                        "Tell me ONE cell like “row 1 column 1 is 5”, or tap a cell and say the digit (1–9)."
            )
        }
    }


    // -------------------------------------------------------------------------
    // FREE-TALK MODE — unchanged
    // -------------------------------------------------------------------------

    override suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): FreeTalkRawResponse = withContext(Dispatchers.IO) {
        // (unchanged from your version)
        try {
            val reqId = UUID.randomUUID().toString().substring(0, 8)
            val t0 = SystemClock.elapsedRealtime()

            fun sha256HexUtf8(s: String): String {
                val md = MessageDigest.getInstance("SHA-256")
                val digest = md.digest(s.toByteArray(Charsets.UTF_8))
                val hex = StringBuilder(digest.size * 2)
                for (b in digest) hex.append(String.format("%02x", b))
                return hex.toString()
            }

            fun capPreview(text: String, maxChars: Int): String {
                val normalized = text
                    .replace('\n', ' ')
                    .replace('\r', ' ')
                    .replace(Regex("\\s+"), " ")
                    .trim()
                if (normalized.length <= maxChars) return normalized
                return normalized.substring(0, maxChars).trimEnd() + "…"
            }

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
                    val m = Regex("^(system|developer|user|assistant)\\s*:\\s?(.*)$", RegexOption.IGNORE_CASE)
                        .find(line)

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

            fun pickSoftRefToken(lastUserText: String): String? {
                val words = lastUserText
                    .lowercase()
                    .replace(Regex("[^a-z0-9\\s]"), " ")
                    .split(Regex("\\s+"))
                    .filter { it.length >= 5 }
                return words.lastOrNull()
            }

            val sysTrim = augmentSystemPrompt(systemPrompt)
            val devTrim = developerPrompt.trim()
            val usrTrim = userMessage.trim()

            val parsed = parseCanonicalHistory(devTrim)
            val budgeted = budgetHistory(parsed.items)

            val pinnedGrid = buildPinnedGridContextSystemMessage(devTrim)
            //val pinnedOrigin = buildPinnedCaptureOriginSystemMessage(devTrim)

            val contextBlob = buildString {
                val pre = parsed.pre.trim().takeIf { it.isNotEmpty() }?.let { capChars(it, 5000) }
                val post = parsed.post.trim().takeIf { it.isNotEmpty() }?.let { capChars(it, 5000) }

                appendLine("BEGIN_CONTEXT")
                if (pre != null) {
                    appendLine(pre); appendLine()
                }
                if (post != null) {
                    appendLine(post); appendLine()
                }

                appendLine("TURN-TAKING / MEMORY CONTRACT (STRICT):")
                appendLine("- The conversation history is provided as actual user/assistant messages below.")
                appendLine("- If the user asks you to repeat what you said earlier, quote the most recent assistant message verbatim from provided history.")
                appendLine("- Do not claim you cannot recall if the answer is present in the provided history.")
                appendLine("END_CONTEXT")
            }.trim()

            val messages = JSONArray().apply {
                put(JSONObject().apply { put("role", "system"); put("content", sysTrim) })
                //put(JSONObject().apply { put("role", "system"); put("content", pinnedOrigin) })
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
            val payloadSha = sha256HexUtf8(payloadStr)

            val lastUserFromHistory = budgeted.lastOrNull { it.first == "user" }?.second ?: ""
            val softToken = pickSoftRefToken(lastUserFromHistory)

            ConversationTelemetry.emitKv(
                "LLM_HTTP_BEGIN",
                "req_id" to reqId,
                "mode" to "FREE_TALK",
                "model" to model,
                "sys_len" to sysTrim.length,
                "dev_len" to devTrim.length,
                "usr_len" to usrTrim.length,
                "dev_has_history_markers" to (devTrim.contains("BEGIN_CANONICAL_HISTORY") && devTrim.contains("END_CANONICAL_HISTORY")),
                "history_parse_ok" to parsed.ok,
                "history_items_total" to parsed.items.size,
                "history_items_budgeted" to budgeted.size,
                "history_user_ct" to parsed.userCount,
                "history_asst_ct" to parsed.assistantCount,
                "payload_bytes" to payloadStr.toByteArray(Charsets.UTF_8).size,
                "payload_sha256" to payloadSha,
                "dev_preview" to capPreview(devTrim, 220),
                "usr_preview" to capPreview(usrTrim, 140),
                "history_last_user_preview" to capPreview(lastUserFromHistory, 140),
                "soft_ref_token" to (softToken ?: JSONObject.NULL)
            )

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
                Log.e("SudokuLLM", "OpenAI freeTalk error: HTTP ${response.code} - $bodyString")
                ConversationTelemetry.emitKv(
                    "LLM_HTTP_END",
                    "req_id" to reqId,
                    "mode" to "FREE_TALK",
                    "model" to model,
                    "http_code" to response.code,
                    "ok" to false,
                    "latency_ms" to dtMs,
                    "body_len" to bodyString.length,
                    "body_preview" to capPreview(bodyString, 260)
                )
                return@withContext FreeTalkRawResponse("I’m having trouble reaching the server. Want to try again?")
            }

            val root = JSONObject(bodyString)

            // ✅ Log the model actually used by the API
            val usedModel = root.optString("model", "<missing>")
            Log.i("SudokuLLM", "OpenAI FREE_TALK response model = $usedModel (requested=$model, req_id=$reqId)")


            val content = root.getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
                .trim()

            val softRefOk = if (!softToken.isNullOrBlank()) content.lowercase().contains(softToken) else JSONObject.NULL

            ConversationTelemetry.emitKv(
                "LLM_HTTP_END",
                "req_id" to reqId,
                "mode" to "FREE_TALK",
                "model" to model,
                "http_code" to response.code,
                "ok" to true,
                "latency_ms" to dtMs,
                "body_len" to bodyString.length,
                "content_len" to content.length,
                "content_sha256" to sha256HexUtf8(content),
                "content_preview" to capPreview(content, 260),
                "soft_ref_ok" to softRefOk
            )

            FreeTalkRawResponse(content)
        } catch (e: Exception) {
            Log.e("SudokuLLM", "Error in freeTalk()", e)
            FreeTalkRawResponse("Something went wrong on my side. Can you repeat that?")
        }
    }

    // -------------------------------------------------------------------------
    // extractClues left as-is
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
                Log.e("SudokuLLM", "OpenAI extractClues error: HTTP ${response.code} - $bodyString")
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