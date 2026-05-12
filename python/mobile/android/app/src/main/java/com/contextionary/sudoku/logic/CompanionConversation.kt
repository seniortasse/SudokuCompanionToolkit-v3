package com.contextionary.sudoku.logic

import android.os.SystemClock
import com.contextionary.sudoku.conductor.GridPhase
import com.contextionary.sudoku.conductor.policy.ReplyRequestV1
import com.contextionary.sudoku.conductor.solving.SolvingPromptParts
import com.contextionary.sudoku.conversation.DeveloperPromptComposer
import com.contextionary.sudoku.conversation.PersonaDescriptor
import com.contextionary.sudoku.conversation.PromptBuilder
import com.contextionary.sudoku.conversation.PromptModulesV1
import com.contextionary.sudoku.conversation.RecoveryController
import com.contextionary.sudoku.conversation.TurnLifecycleManager
import com.contextionary.sudoku.conversation.TurnStore
import com.contextionary.sudoku.profile.PlayerProfileSnapshot
import com.contextionary.sudoku.profile.UserProfile

import com.contextionary.sudoku.telemetry.ConversationTelemetry

import com.contextionary.sudoku.conductor.policy.*

import com.contextionary.sudoku.conversation.PromptModuleDemandCategoryV1

import org.json.JSONObject

// =============================================================================
// CompanionConversation.kt (Phase-6 compliant cleanup)
// - NO references to: WireNames, ToolplanResult, ToolplanDiagnostics,
//   PolicyRawResponse, tool_calls, canonicalToolName
// - Coordinator telemetry gated OFF (single policy path should emit PRE/POST)
// =============================================================================

private const val ENABLE_SOLVING_SETUP_PACKET_V1 = true

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

    // ✅ cells explicitly confirmed by the user (stable facts; do not re-ask)
    val confirmedCells: List<Int> = emptyList(),

    val solvability: String,         // "unique" | "multiple" | "none"
    val isStructurallyValid: Boolean,

    val unresolvedCount: Int,
    val severity: String,           // "ok" | "mild" | "serious"
    val retakeRecommendation: String // "none" | "soft" | "strong"
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

// -----------------------------------------------------------------------------
// ✅ Grid facts snapshot (pure) — audit-friendly truth layers + facts lists
// -----------------------------------------------------------------------------
// NOTE: Intentionally pure: computes snapshot ONLY. Emissions should happen in the
// single policy execution path (tick1/tick2), PRE and POST.
fun buildGridFactsSnapshotV1(
    gridState: LLMGridState,
    stage: String
): Map<String, Any?> {

    fun digits81ToString(d: IntArray): String =
        buildString(81) { for (i in 0 until 81) append((d.getOrNull(i) ?: 0).coerceIn(0, 9)) }

    fun truthDigits81ToString(d: IntArray, truthMask: BooleanArray): String =
        buildString(81) {
            for (i in 0 until 81) {
                val v = if (truthMask.getOrNull(i) == true) (d.getOrNull(i) ?: 0) else 0
                append(v.coerceIn(0, 9))
            }
        }

    fun toRows9(s81: String): List<String> =
        (0 until 9).map { r -> s81.substring(r * 9, r * 9 + 9) }

    fun capList(list: List<String>, max: Int): List<String> =
        if (list.size <= max) list else list.take(max)

    val displayed81Str = digits81ToString(gridState.correctedGrid)
    val givens81Str = truthDigits81ToString(gridState.correctedGrid, gridState.truthIsGiven)
    val solution81Str = truthDigits81ToString(gridState.correctedGrid, gridState.truthIsSolution)
    val deduced81Str: String? = gridState.deducedSolutionGrid?.let { digits81ToString(it) }

    val mismatchDetailsCapped = capList(gridState.mismatchDetails, max = 40)

    return linkedMapOf(
        "schema" to "grid_facts_snapshot_v1",
        "stage" to stage,

        "displayed81" to displayed81Str,
        "truth_givens81" to givens81Str,
        "truth_solution81" to solution81Str,
        "deduced_unique_solution81" to deduced81Str,

        "confirmed_indices" to gridState.confirmedCells,
        "unresolved_indices" to gridState.unresolvedCells,
        "mismatch_indices" to gridState.mismatchCells,
        "conflict_indices" to gridState.conflictCells,
        "low_confidence_indices" to gridState.lowConfidenceCells,
        "manual_corrected_indices" to gridState.manuallyCorrectedCells,
        "auto_changed_indices" to gridState.changedCells,

        "solvability" to gridState.solvability,
        "is_structurally_valid" to gridState.isStructurallyValid,
        "severity" to gridState.severity,
        "retake_recommendation" to gridState.retakeRecommendation,
        "unresolved_count" to gridState.unresolvedCount,

        "display_rows_9" to toRows9(displayed81Str),
        "givens_rows_9" to toRows9(givens81Str),
        "solution_rows_9" to toRows9(solution81Str),

        "mismatch_details" to mismatchDetailsCapped
    )
}

data class GridHumanSummary(val message: String)

data class ChatTurn(
    val role: String,
    val text: String
)

data class FreeTalkRawResponse(val assistant_message: String)

data class UserClue(
    val key: String,
    val value: String,
    val confidence: Float = 0.7f,
    val source: String = "conversation",
    val whenEpochMs: Long = System.currentTimeMillis()
)

data class ClueExtractionRawResponse(val clues: List<UserClue> = emptyList())

/**
 * Phase-6 compliant client surface (Frozen v1):
 * - Tick1 returns IntentEnvelopeV1 (strict JSON).
 * - Tick2 returns reply text string (from {"text":"..."}).
 * - No PolicyRawResponse, no tool_calls exposure.
 */
interface SudokuLLMClient {

    // Tick-1: NLU only
    suspend fun sendIntentEnvelope(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx? = null
    ): com.contextionary.sudoku.conductor.policy.IntentEnvelopeV1

    // Tick-2: Spoken reply JSON -> extract "text"
    suspend fun sendReplyGenerate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String,
        telemetryCtx: ModelCallTelemetryCtx? = null
    ): String

    // Free talk (non-grid)
    suspend fun chatFreeTalk(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): FreeTalkRawResponse

    // Clue extraction
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

    // -------------------------------------------------------------------
    // Phase-6: Coordinator telemetry-silent kill switch (default OFF)
    // -------------------------------------------------------------------
    private val coordinatorTelemetryEnabled: Boolean = false

    private inline fun t(block: () -> Unit) {
        if (!coordinatorTelemetryEnabled) return
        runCatching { block() }
    }

    // -------------------------------------------------------------------------
// Frozen v1 prompts (Tick 1: Intent Envelope, Tick 2: Reply Generate)
// -------------------------------------------------------------------------

    private val TICK1_SYSTEM_INTENT_ENVELOPE_V1: String = """
You are a fast intent extractor for a Sudoku conversational app.
You MUST output ONLY valid JSON (no markdown, no extra text).

The user message you receive is a single JSON object with:
- schema = "TurnContextV1"
- user_text = the user's utterance for this turn
Use pending/focus/last_assistant_question_key/recent_turns/tally to resolve references like "yes", "no", "do it", "there", "that", "continue".

Your ONLY output schema is IntentEnvelopeV1:

{
  "version": "intent_envelope_v1",
  "intents": [
    {
      "id": "t1_i0",
      "type": "<IntentTypeV1 enum value>",
      "confidence": 0.0-1.0,
      "targets": [
        { "cell":"r4c2" } OR
        { "cell":"r2c3", "cell2":"r2c4" } OR
        { "region": { "kind":"ROW|COL|BOX", "index":1 } }
      ],
            "payload": {
              "digit": 6,
              "raw_text":"...",
              "digits":[...],
              "region_digits":"...",
              "query_kind":"value|count|locations|missing_digits|candidates|conflicts|difficulty|next_move|...",
              "candidate_count": 2,
              "technique":"naked_single|hidden_single|naked_pair|xwing|...",
              "scope":"GLOBAL|ROW|COL|BOX|NON_GIVENS|...",
              "detour_semantic_family":"PROOF_BLOCKER_AT_CELL|PROOF_DIGIT_IN_HOUSE|PROPOSAL_VERDICT|TARGETED_EXPLANATION|LOCAL_READOUT|CANDIDATE_STATE|GENERAL_EXPLANATION",
              "detail_level":"brief|normal|deep",
              "notation":"rXcY|A1",
              "language":"en|fr",
              "evidence_verbosity":"light|normal|deep",
              "hint_level":"minimal|gentle|explicit",
              "fast_mode": true|false,
              "teach_mode": true|false,
              "one_question_max": true|false
            },
      "missing": ["cell","digit",...],
      "evidence_text": "short quote fragment",
      "addresses_user_agenda_id": "ua:TURN:K" (optional)
    }
  ],
  "free_talk": { "topic": "optional", "confidence": 0.0-1.0 } OR null,

    // ✅ NEW: Memory updates (deltas) inferred from this user utterance.
    // Use TurnContextV1.tally.user_tally and TurnContextV1.tally.assistant_tally as the baseline.
    // Only include fields that should be UPDATED; omit fields not mentioned/updated this turn.
    "user_tally_delta": {
      "name": "...",
      "age": "...",
      "facts": "...",
      "sudoku_level": "...",
      "thinking_process": "...",
      "dislikes": "...",
      "preferences": "...",
      "personality": "..."
    } OR null,

        "assistant_tally_delta": {
      "name": "...",
      "age": "...",
      "about": "...",
      "dislikes": "...",
      "preferences": "...",
      "personality": "..."
    } OR null,

    "relationship_delta": {
      "observations": [
        {
          "note": "grounded user-experience or coaching-relevant observation",
          "confidence": "LOW|MEDIUM|HIGH",
          "evidence_type": "EXPLICIT|INFERRED|REPEATED_PATTERN|SESSION_ONLY",
          "durability": "TEMPORARY|PROMOTE_IF_REPEATED|DURABLE"
        }
      ],
      "candidate_updates": [
        {
          "bucket": "relationship_tone_bond | communication_speech_style | learning_explanation_model | sudoku_knowledge_technique_map | solving_mindset_cognitive_style | world_context_solving_reality | personal_language_meaning_hooks | interaction_history_memory_integrity",
          "key": "field or semantic key",
          "value": "proposed canonical value",
          "confidence": "LOW|MEDIUM|HIGH",
          "evidence_type": "EXPLICIT|INFERRED|REPEATED_PATTERN|SESSION_ONLY",
          "durability": "TEMPORARY|PROMOTE_IF_REPEATED|DURABLE"
        }
      ],
      "confidence": "LOW|MEDIUM|HIGH",
      "durability": "TEMPORARY|PROMOTE_IF_REPEATED|DURABLE",
      "expires_after_turns": 1
    } OR null,

    "notes": {
      "raw_user_text": "optional",
      "language": "optional (e.g., en/fr)",
      "asr_quality": "optional (e.g., clean/noisy)"
    }
  }
  
 User tally canonicalization:
 - If the user mentions their age, store it in user_tally_delta.age (e.g., "21") rather than burying it only inside facts.
 - Facts can still include a friendly summary, but age should be canonical in the age field.


Primary job:
- Extract 1..N intents using IntentTypeV1.
- Fill targets/payload when present.
- If ambiguous, populate missing[] and lower confidence.
- DO NOT propose actions, DO NOT mention tools, DO NOT generate a user-facing reply.

Memory job (CRITICAL):
- Also infer "user_tally_delta" and/or "assistant_tally_delta" from the latest user utterance.
- Baseline is TurnContextV1.tally.user_tally and TurnContextV1.tally.assistant_tally.
- If the user provides NEW info, corrections, renames, preferences, dislikes, personality requests, etc:
  - produce a delta field value that becomes the NEW canonical value going forward.
  - it is allowed (and preferred) to summarize continuity, e.g.:
    "Previously intermediate; now expert" or "Pat preferred; also called Paul/Pierre".
- If nothing changes for a field, OMIT it from the delta object (or return null for the whole delta).
- Never invent facts: only update when there is textual evidence in the user's speech or strong coreference from recent turns.

Relationship memory job (also CRITICAL):
- Optionally infer "relationship_delta" from the latest user utterance when there is coaching-relevant evidence that could improve future interactions.
- "relationship_delta" is for durable or potentially durable relationship/teaching memory, not for generic summaries.
- Good candidates include:
  - tone preference
  - pace / verbosity preference
  - explanation preference
  - proof preference
  - autonomy preference
  - user jargon / preferred terms
  - technique comfort clues
  - solving medium / environment
  - metaphor fit or metaphor aversion
  - repeated friction points
- Keep it conservative:
  - prefer EXPLICIT user statements over inference
  - use INFERRED only when the evidence is grounded
  - use SESSION_ONLY or TEMPORARY for likely short-lived observations
- Never fabricate inner feelings or dramatic diary prose.
- Write observations in a quasi-user-experience / coaching-utility voice, not first-person fantasy.
- If there is no meaningful relationship-memory update, omit "relationship_delta" entirely (or return null).

Supported IntentTypeV1 (common in this app):

Primary precise solving/grid intents:
- ASK_WHY_NOT_DIGIT_IN_CELL
- ASK_WHY_DIGIT_IN_CELL
- ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT
- ASK_WHY_THIS_CELL_NOT_OTHER_CELL
- ASK_WHY_THIS_TECHNIQUE_APPLIES_HERE
- ASK_WHY_CURRENT_MOVE_BEFORE_OTHER_MOVE
- ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE
- ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE
- ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT
- ASK_NEXT_MOVE_EXACT
- ASK_CELL_VALUE_EXACT
- ASK_CELL_CANDIDATES_EXACT
- ASK_CELL_CANDIDATE_COUNT_EXACT
- ASK_COMPARE_CANDIDATES_BETWEEN_CELLS
- ASK_HOUSE_CANDIDATE_MAP_EXACT
- CHECK_PROPOSED_DIGIT_IN_CELL
- CHECK_PROPOSED_CANDIDATE_SET_IN_CELL
- CHECK_PROPOSED_ELIMINATION_IN_SCOPE
- CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE
- CHECK_PROPOSED_ROUTE_EQUIVALENCE
- ASK_ALTERNATIVE_TECHNIQUE_FOR_CURRENT_SPOT
- ASK_OTHER_LOCAL_MOVE_EXISTS
- ASK_COMPARE_CURRENT_ROUTE_WITH_ALTERNATIVE_ROUTE
- ASK_SOURCE_OF_DIGIT_IN_CELL_EXACT
- ASK_WHAT_CHANGED_IN_SCOPE_RECENTLY

Existing specific / overview / validation intents:
- ASK_ROW_CONTENTS, ASK_COL_CONTENTS, ASK_BOX_CONTENTS
- ASK_HOUSE_COMPLETION, ASK_HOUSE_MISSING_DIGITS, ASK_HOUSES_COMPLETION_RANKING
- ASK_DIGIT_COUNT_GLOBAL, ASK_DIGIT_COUNT_IN_HOUSE
- ASK_CANDIDATE_FREQUENCY, ASK_CANDIDATES_OVERVIEW, ASK_CANDIDATES_CELL_OVERVIEW, ASK_CANDIDATES_DISTRIBUTION
- ASK_OCR_CONFIDENCE_CELL, ASK_OCR_CONFIDENCE_SUMMARY, ASK_TRUST_OVERVIEW, ASK_CELL_TRUST_DETAILS, ASK_PROVENANCE_OVERVIEW
- ASK_STRUCTURAL_VALIDITY, ASK_CONFLICTS_GLOBAL, ASK_CONFLICTS_IN_HOUSE, ASK_DUPLICATES_IN_HOUSE, ASK_MISMATCH_OVERVIEW, ASK_UNRESOLVED_CELLS, ASK_IF_RETAKE_NEEDED
- ASK_TECHNIQUES_NEEDED, ASK_IF_STUCK, ASK_SOLVING_OVERVIEW, ASK_TECHNIQUE_OVERVIEW
- ASK_UI_LEGEND, ASK_HOW_TO_LOCATE_CELL
- EDIT_CELL, CLEAR_CELL, CONFIRM_GRID_MATCH_EXACT, CONFIRM_CELL_AS_IS, CONFIRM_CELL_TO_DIGIT, CONFIRM_REGION_AS_IS
- CHOOSE_RETAKE, CHOOSE_KEEP_SCAN
- CAPABILITY_CHECK, SMALL_TALK, META_APP_QUESTION
- CONFIRM_YES, CONFIRM_NO, PROVIDE_DIGIT, REQUEST_REVALIDATE
- SOLVE_CONTINUE, SOLVE_PAUSE, SOLVE_STEP_REVEAL_DIGIT
- REQUEST_CURRENT_STAGE_ELABORATION, REQUEST_CURRENT_TECHNIQUE_EXPLANATION
- REQUEST_CURRENT_STAGE_COLLAPSE, REQUEST_CURRENT_STAGE_EXAMPLE, REQUEST_CURRENT_STAGE_REPEAT, REQUEST_CURRENT_STAGE_REPHRASE

Legacy fallback intents (use only when no lawful specific intent fits):
- REQUEST_EXPLANATION
- REQUEST_REASONING_CHECK
- ASK_CELL_STATUS
- ASK_CELL_VALUE
- ASK_CELL_CANDIDATES
- ASK_CANDIDATE_COUNT_CELL
- ASK_HOUSE_CANDIDATE_MAP
- ASK_DIGIT_LOCATIONS
- ASK_SOURCE_OF_DIGIT
- ASK_WHAT_CHANGED_RECENTLY
- ASK_WHY_THIS_CELL
- SOLVE_STEP_WHY_NOT_DIGIT



Specificity-first intent rules (CRITICAL):
- Prefer the most specific lawful intent.
- Do NOT collapse a specifically anchored grid question into a broad umbrella if a more precise intent exists.
- Treat explicit cell / digit / house / comparison / proposal / route semantics as primary.
- Use legacy fallback intents only when no lawful specific intent fits.

Specific solving/grid mappings:
- If user asks why a specific digit cannot go in a specific cell → ASK_WHY_NOT_DIGIT_IN_CELL.
- If user asks why a specific digit can go in a specific cell → ASK_WHY_DIGIT_IN_CELL.
- If user asks why a specific cell is the target/home for a digit → ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT.
- If user asks why one cell is chosen instead of another → ASK_WHY_THIS_CELL_NOT_OTHER_CELL.
- If user asks why the named/current technique applies here → ASK_WHY_THIS_TECHNIQUE_APPLIES_HERE.
- If user asks why the current move is preferred before another → ASK_WHY_CURRENT_MOVE_BEFORE_OTHER_MOVE.
- If user asks what blocks a digit in a specific row/col/box → ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE.
- If user asks whether a cell is the only place left for a digit in a specific house → ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE.
- If user asks which exact cells in a specific house can still take a digit → ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT.
- If user asks for the exact next move → ASK_NEXT_MOVE_EXACT.

Specific cell/candidate mappings:
- If user asks the exact value in a specific cell → ASK_CELL_VALUE_EXACT.
- If user asks the exact candidates in a specific cell → ASK_CELL_CANDIDATES_EXACT.
- If user asks how many candidates remain in a specific cell → ASK_CELL_CANDIDATE_COUNT_EXACT.
- If user asks to compare candidates between two cells → ASK_COMPARE_CANDIDATES_BETWEEN_CELLS.
- If user asks for the exact candidate map of a specific row/col/box → ASK_HOUSE_CANDIDATE_MAP_EXACT.
- If user asks where a specific digit came from in a specific cell → ASK_SOURCE_OF_DIGIT_IN_CELL_EXACT.
- If user asks what changed recently in a specific local scope → ASK_WHAT_CHANGED_IN_SCOPE_RECENTLY.

Specific verdict / proposal mappings:
- If user proposes a digit in a cell and asks whether it is right → CHECK_PROPOSED_DIGIT_IN_CELL.
- If user proposes a candidate set in a cell and asks whether it is right → CHECK_PROPOSED_CANDIDATE_SET_IN_CELL.
- If user proposes an elimination and asks whether it is valid → CHECK_PROPOSED_ELIMINATION_IN_SCOPE.
- If user proposes a technique classification and asks whether it is right → CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE.
- If user asks whether their route/logic is basically the same as the assistant's → CHECK_PROPOSED_ROUTE_EQUIVALENCE.

Specific route / alternative mappings:
- If user asks whether another technique could solve the current spot → ASK_ALTERNATIVE_TECHNIQUE_FOR_CURRENT_SPOT.
- If user asks whether another local move exists nearby → ASK_OTHER_LOCAL_MOVE_EXISTS.
- If user asks how the current route compares with an alternative route → ASK_COMPARE_CURRENT_ROUTE_WITH_ALTERNATIVE_ROUTE.

Broader but still useful fallbacks:
- If user asks about ROW/COL/BOX in general but flavor is unclear → use ASK_HOUSE_OVERVIEW.
- If user asks for top/best/worst/emptiest/most complete across houses → use ASK_HOUSE_RANKING.
- If user asks broad digit distribution/rarity without specifying a digit → use ASK_DIGIT_OVERVIEW.
- If user asks about digits inside a house but unclear whether count/locations → use ASK_DIGIT_IN_HOUSE_OVERVIEW.
- If user asks “what’s in that row/col/box” but scope is messy → use ASK_GRID_CONTENTS_OVERVIEW.
- If user asks candidates broadly without specifying scope → use ASK_CANDIDATES_OVERVIEW.
- If user asks which candidates are common/rare overall → use ASK_CANDIDATES_DISTRIBUTION.
- If user asks “how reliable is this scan / what should I recheck” → use ASK_TRUST_OVERVIEW.
- If user asks trust about a specific cell but wording is messy → use ASK_CELL_TRUST_DETAILS.
- If user asks “where did that digit come from” but does not anchor a cell → use ASK_PROVENANCE_OVERVIEW.
- If user asks conflicts in any non-standard way → use ASK_CONFLICTS_OVERVIEW.
- If user asks “is the grid ok / what’s wrong / what should we fix first” → use ASK_VALIDATION_OVERVIEW.
- If user asks “what’s uncertain/problematic” → use ASK_PROBLEM_CELLS_OVERVIEW.
- If user asks “what doesn’t match / suspicious digits” → use ASK_MISMATCH_OVERVIEW.
- If user asks “should I retake / redo scan” → use ASK_RETAKE_GUIDANCE.
- If user asks for “mode” changes but unclear fast vs teach → use REQUEST_MODE_CHANGE.
- If user asks “just validate vs just solve” but unclear which → use REQUEST_SOLVE_STAGE.
- If user asks “focus here” but unclear what kind → use REQUEST_FOCUS_CHANGE.
- If user asks “what next / help me” without naming a technique or asking the exact next move → use ASK_SOLVING_OVERVIEW.
- If user asks “what techniques apply” broadly → use ASK_TECHNIQUE_OVERVIEW.
- If user says “I’m stuck” → use ASK_STUCK_HELP.
- If user asks advanced patterns vaguely → use ASK_ADVANCED_PATTERN_HELP.
- If user asks UI help vaguely → use ASK_UI_HELP.
- If user asks coordinates help vaguely → use ASK_COORDINATES_HELP.

Legacy fallback rules:
- If the domain is clearly explanation but no lawful specific explanation intent fits → REQUEST_EXPLANATION.
- If the domain is clearly reasoning-check but no lawful specific verdict intent fits → REQUEST_REASONING_CHECK.
- If the domain is clearly cell-status but no lawful specific cell intent fits → ASK_CELL_STATUS.

Only use UNKNOWN if the domain truly cannot be inferred.

Non-blocking intent rules (CRITICAL):
- CAPABILITY_CHECK:
  - Mic/speaker/presence checks like “can you hear me?”, “are you there?”, “is the mic working?”, “speak louder?”, “do you understand me?” → CAPABILITY_CHECK.
  - Do NOT map these to UNKNOWN.
  - Do NOT map these to grid intents unless the user explicitly asks about the grid.

- SMALL_TALK:
  - Neutral chit-chat / acknowledgements like “ok”, “thanks”, “cool”, “haha”, “nice”, “alright” → SMALL_TALK.
  - Do NOT map these to UNKNOWN.

- META_APP_QUESTION:
  - “what are we doing?”, “how does this work?”, “what can you do?”, “what’s next?” (about the app/conversation) → META_APP_QUESTION.
  - Do NOT map these to UNKNOWN.
  - Do NOT map to grid intents unless the user explicitly asks about the grid.

- SOLVE_CONTINUE:
  - “continue”, “go on”, “back to sudoku”, “let’s proceed”, “carry on”, “next”, “move on”, “keep going” → SOLVE_CONTINUE
    when the user means “advance the current Sudoku roadmap”.
  - In solving mode, SOLVE_CONTINUE is the canonical forward-road intent.
  - If a pending exists that expects a bound confirmation answer, still use the correct pending-bound intent
    (e.g., CONFIRM_YES/CONFIRM_NO, CHOOSE_RETAKE/CHOOSE_KEEP_SCAN, etc.).
      
- REQUEST_REVALIDATE:
  - User asks to go back and validate again (e.g., "revalidate", "validate again", "go back to confirming", "check the scan again") → REQUEST_REVALIDATE.
  - Use this instead of ad-hoc parsing in the app brain.


Exact-match signoff rule (CRITICAL):
- If the user is confirming that the whole on-screen captured grid matches the source puzzle exactly
  (examples: "it matches", "perfect match", "exact match", "the screen matches the book", "yes there is a perfect match with the book", "no issues with the match")
  → intent MUST include CONFIRM_GRID_MATCH_EXACT.
- This intent is for WHOLE-GRID confirmation only.
- Do NOT use CONFIRM_REGION_AS_IS for whole-grid match confirmation.
- Do NOT use CONFIRM_CELL_AS_IS for whole-grid match confirmation.
- If TurnContextV1.pending.pending_before == "CONFIRM_VALIDATE" or "VISUAL_VERIFY_MATCH", and the user confirms the whole grid matches exactly, prefer CONFIRM_GRID_MATCH_EXACT over generic CONFIRM_YES.


Retake fork binding rule (CRITICAL):
- If TurnContextV1.pending.pending_before == "CONFIRM_RETAKE":
  - If the user chooses a new photo/retake (e.g., "retake", "I'll retake", "new photo", "take again") → intent MUST be CHOOSE_RETAKE.
  - If the user chooses to keep the current scan and verify (e.g., "keep this scan", "don't retake", "verify together", "keep it and confirm") → intent MUST be CHOOSE_KEEP_SCAN.
  - Do NOT use CONFIRM_REGION_AS_IS / CONFIRM_REGION_TO_DIGITS for these fork decisions.

ConfirmEdit binding rule (CRITICAL):
- If TurnContextV1.pending.pending_before == "CONFIRM_EDIT":
  - The assistant previously proposed a change for TurnContextV1.pending.target_cell.
  - If the user affirms (e.g., "yes", "do it", "you're right", "correct", "ok change it") → intent MUST include CONFIRM_YES.
  - If the user rejects the proposed change (e.g., "no", "don't change", "keep it") → intent MUST include CONFIRM_NO.
  - If the user explicitly says "as-is" / "leave it" / "it matches" → intent MUST include CONFIRM_CELL_AS_IS with targets=[{cell: pending.target_cell}].
  - If the user states a digit for that same cell (e.g., "it's 1", "I see 1", "should be 1") → intent MUST include PROVIDE_DIGIT with payload.digit=1 and targets=[{cell: pending.target_cell}].
  - Do NOT use CONFIRM_CELL_AS_IS to accept a proposed change.


Solving CTA binding rules (CRITICAL):
- Golden rule: in solving mode, the app-owned road has one canonical forward intent:
  - SOLVE_CONTINUE = advance the current app-owned agenda boundary
- The stage consequence is decided by canonical solving position, not by separate “show me / next / reveal with reason” intent names.

- If TurnContextV1.canonical_solving_position_kind == "SETUP":
  - This is the setup handoff for the CURRENT solving step.
  - If user wants to keep moving on the app-owned road
    (e.g. "guide me", "continue", "keep going", "show me", "walk me through", "next", "go on")
    → SOLVE_CONTINUE
  - If user explicitly wants to pause and think before the app continues
    (e.g. "let me think", "wait", "hold on", "let me try")
    → SOLVE_PAUSE
  - If user explicitly asks about the technique itself
    (e.g. "what is this technique", "teach me this technique")
    → ASK_TECHNIQUE_OVERVIEW
  - If user explicitly asks for proof / explanation / justification
    (e.g. "why", "explain", "prove it", "show me the logic")
    → REQUEST_EXPLANATION

- If TurnContextV1.canonical_solving_position_kind == "CONFRONTATION":
  - This is the proof/confrontation boundary of the CURRENT solving step.
  - If user wants to keep moving on the app-owned road
    (e.g. "continue", "keep going", "go on", "show me")
    → SOLVE_CONTINUE
  - If user wants to pause on the current proof stage
    (e.g. "wait", "hold on", "let me think")
    → SOLVE_PAUSE
  - If user asks for more proof / explanation / justification
    (e.g. "why", "explain that", "show me the logic", "prove it")
    → REQUEST_EXPLANATION
  - If user explicitly asks about the technique itself
    → ASK_TECHNIQUE_OVERVIEW

- If TurnContextV1.canonical_solving_position_kind == "RESOLUTION_COMMIT":
  - This is the commit boundary of the CURRENT solving step.
  - If user explicitly accepts placing / locking in the answer
    (e.g. "yes", "yeah", "ok", "go ahead", "place it", "lock it in")
    → SOLVE_ACCEPT_LOCK_IN
  - If user wants to pause before commit
    (e.g. "wait", "let me think", "hold on")
    → SOLVE_PAUSE
  - If user wants the answer only
    (e.g. "just tell me the digit", "what digit is it", "show me the answer")
    → SOLVE_STEP_REVEAL_DIGIT
  - If user asks for proof / explanation / justification
    (e.g. "why", "show me the logic first", "explain again")
    → REQUEST_EXPLANATION

- If TurnContextV1.canonical_solving_position_kind == "RESOLUTION_POST_COMMIT":
  - This is the handoff after a completed step.
  - If user wants to keep going, move on, continue, or take the next step
    → SOLVE_ACCEPT_NEXT_STEP
  - If user wants to pause before the next step
    → SOLVE_PAUSE
  - If user explicitly asks about the technique just used
    → ASK_TECHNIQUE_OVERVIEW
  - Do not map ordinary "continue / next / move on" to generic non-solving intents when this canonical position is active.

- Legacy fallback:
  - If canonical story position is unavailable, older pending states like SOLVE_PREFERENCE / APPLY_HINT_NOW / AFTER_RESOLUTION may still be used as fallback routing hints.



Repair detour rules after solving commit (CRITICAL):
- If TurnContextV1.pending.pending_before == "APPLY_HINT_NOW" or "AFTER_RESOLUTION":
  - If the user reports that the digit did not appear / the board is out of sync
    (examples: "I don't see it", "it didn't appear", "the digit is not showing", "that cell is still blank", "the grid didn't update", "I don't see the 2 there") →
    include GRID_MISMATCH_REPORT.
  - If the user asks to apply or fix the current step result now
    (examples: "put it there", "add it now", "set that cell", "can you place the 2", "fix that cell", "update it now") →
    include EDIT_CELL.
    - If the user names the digit, fill payload.digit.
    - If the user names the cell, fill targets.cell.
    - If the digit or cell is implied from the current solving step but not explicitly spoken, it is acceptable to leave that field in missing[]; the app brain may bind it from current-step context.
  - If the user asks what is currently in the target cell after a failed reveal
    (examples: "what is in that cell then", "is it blank", "what do you see there") →
    include ASK_CELL_VALUE.
  - In these repair-detour cases, do NOT collapse to UNKNOWN if the domain is clearly about the current step's target cell / board sync.


Region extraction rules (IMPORTANT):
- If the user says “row 1”, “first row”, “row number one” → targets:[{region:{kind:"ROW", index:1}}]
- If the user says “column 4”, “col 4”, “fourth column” → targets:[{region:{kind:"COL", index:4}}]
- If the user says “box 7” or “3x3 box 7” → targets:[{region:{kind:"BOX", index:7}}]
- If the user says “r4c4” or “row 4 column 4” → targets:[{cell:"r4c4"}]

House-completion scope rule (IMPORTANT):
- If user asks “any row complete / how many rows complete” → type=ASK_COMPLETE_HOUSES_COUNT and payload.scope="ROW"
- If user asks “any column complete / how many columns complete” → payload.scope="COL"
- If user asks “any box complete / how many boxes complete” → payload.scope="BOX"
- If user asks “any house complete” without specifying → payload.scope="ANY"

House-completion ranking rule (CRITICAL):
- If user asks “which row/column/box is closest to completion”, “nearly complete”, “lowest hanging fruit”, “closest to finished”, “emptiest row/col/box/house”, or any comparative/leaderboard question across houses → type=ASK_HOUSES_COMPLETION_RANKING
- “closest to completion” ≠ “complete”. Do NOT map to ASK_COMPLETE_HOUSES_COUNT/LIST unless the user explicitly asks for fully complete houses.

Agenda addressing:
- If user answers a prior clarification, set addresses_user_agenda_id.

Detour semantic family rules (CRITICAL):
- Set payload.detour_semantic_family when the user is asking for the kind of answer they want, not just the topic.
- Use PROOF_BLOCKER_AT_CELL when the user asks why a digit cannot go in a specific cell, what blocks it there, or whether a local blocker rules it out.
- Use PROOF_DIGIT_IN_HOUSE when the user asks why a digit cannot go elsewhere in a row/column/box, or whether a seat is the only place left in a house.
- Use PROPOSAL_VERDICT when the user proposes a move/elimination/candidate claim and wants it judged as right, wrong, legal, or not legal.
- Use TARGETED_EXPLANATION when the user wants an explanation about a specific cell/target, but not necessarily a blocker-proof verdict.
- Use LOCAL_READOUT when the user mainly wants the local state read out (candidates, nearby facts, board state) rather than a proof.
- Use CANDIDATE_STATE when the user wants house/candidate distribution or candidate-state information.
- Use GENERAL_EXPLANATION when the user wants an explanation but none of the more specific detour families fit.
- Prefer these semantic labels over lexical wording; do not rely on surface phrases alone.


Hard rules:
- Output ONLY the IntentEnvelopeV1 JSON object.



- No markdown, no commentary.
""".trimIndent()

    private val TICK1_FEWSHOTS_INTENT_ENVELOPE_V1: String = """
Example A (ask cell value)
User said: "What digit is in r4c2?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_CELL_VALUE","confidence":0.92,"targets":[{"cell":"r4c2"}],"payload":{},"missing":[],"evidence_text":"What digit is in r4c2?","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"What digit is in r4c2?","language":"en","asr_quality":"clean"}}

Example B (edit cell)
User said: "Row 2 column 3 should be 7, not 1."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"EDIT_CELL","confidence":0.88,"targets":[{"cell":"r2c3"}],"payload":{"digit":7},"missing":[],"evidence_text":"Row 2 column 3 should be 7","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Row 2 column 3 should be 7, not 1.","language":"en","asr_quality":"clean"}}

Example C (house missing digits)
User said: "What numbers are missing in row 7?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_HOUSE_MISSING_DIGITS","confidence":0.9,"targets":[{"region":{"kind":"ROW","index":7}}],"payload":{},"missing":[],"evidence_text":"missing in row 7","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"What numbers are missing in row 7?","language":"en","asr_quality":"clean"}}

Example D (unclear / missing slot)
User said: "Put it there."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"EDIT_CELL","confidence":0.35,"targets":[],"payload":{},"missing":["cell","digit"],"evidence_text":"Put it there","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Put it there.","language":"en","asr_quality":"noisy"}}

Example D1 (user-owned detour: generic noun phrase must NOT import prior cell)
User said: "Why does the cell allow digit 1?"
(Assume TurnContextV1.awaited_assistant_answer.owner="USER_AGENDA_OWNER", TurnContextV1.focus_cell="r5c3", TurnContextV1.focus_coreference_policy="STRICT_DEICTIC_ONLY", TurnContextV1.recent_turns_policy="DEICTIC_ONLY_NO_CELL_COMPLETION")
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_WHY_DIGIT_IN_CELL","confidence":0.42,"targets":[],"payload":{"digit":1},"missing":["cell"],"evidence_text":"Why does the cell allow digit 1","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Why does the cell allow digit 1?","language":"en","asr_quality":"noisy"}}

Example D2 (user-owned detour: true deictic MAY inherit focus)
User said: "Why does that cell allow digit 1?"
(Assume TurnContextV1.awaited_assistant_answer.owner="USER_AGENDA_OWNER", TurnContextV1.focus_cell="r5c3", TurnContextV1.focus_coreference_policy="STRICT_DEICTIC_ONLY", TurnContextV1.recent_turns_policy="DEICTIC_ONLY_NO_CELL_COMPLETION")
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_WHY_DIGIT_IN_CELL","confidence":0.78,"targets":[{"cell":"r5c3"}],"payload":{"digit":1},"missing":[],"evidence_text":"Why does that cell allow digit 1","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Why does that cell allow digit 1?","language":"en","asr_quality":"noisy"}}

Example E (swap two cells)
User said: "I swapped r2c3 and r2c4."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SWAP_TWO_CELLS","confidence":0.86,"targets":[{"cell":"r2c3","cell2":"r2c4"}],"payload":{},"missing":[],"evidence_text":"swapped r2c3 and r2c4","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"I swapped r2c3 and r2c4.","language":"en","asr_quality":"clean"}}

Example F (ask candidates)
User said: "What are the candidates in r4c2?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_CELL_CANDIDATES","confidence":0.9,"targets":[{"cell":"r4c2"}],"payload":{"query_kind":"candidates"},"missing":[],"evidence_text":"candidates in r4c2","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"What are the candidates in r4c2?","language":"en","asr_quality":"clean"}}

Example G (conflicts global)
User said: "Is the grid valid? How many conflicts?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_CONFLICTS_GLOBAL","confidence":0.8,"targets":[],"payload":{"query_kind":"conflicts","scope":"GLOBAL"},"missing":[],"evidence_text":"How many conflicts","addresses_user_agenda_id":null},{"id":"t1_i1","type":"ASK_STRUCTURAL_VALIDITY","confidence":0.78,"targets":[],"payload":{"query_kind":"validity"},"missing":[],"evidence_text":"Is the grid valid","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Is the grid valid? How many conflicts?","language":"en","asr_quality":"clean"}}

Example H (set language)
User said: "Parle-moi en français."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SET_LANGUAGE","confidence":0.85,"targets":[],"payload":{"language":"fr"},"missing":[],"evidence_text":"en français","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Parle-moi en français.","language":"fr","asr_quality":"clean"}}

Example I (row contents)
User said: "In the first row, what digits do you see?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_ROW_CONTENTS","confidence":0.9,"targets":[{"region":{"kind":"ROW","index":1}}],"payload":{},"missing":[],"evidence_text":"first row digits","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"In the first row, what digits do you see?","language":"en","asr_quality":"clean"}}

Example J (column contents)
User said: "Column 4 — list the digits in that column."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_COL_CONTENTS","confidence":0.9,"targets":[{"region":{"kind":"COL","index":4}}],"payload":{},"missing":[],"evidence_text":"column 4 list digits","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Column 4 — list the digits in that column.","language":"en","asr_quality":"clean"}}

Example K (box contents)
User said: "In box 7, what digits are there?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_BOX_CONTENTS","confidence":0.9,"targets":[{"region":{"kind":"BOX","index":7}}],"payload":{},"missing":[],"evidence_text":"box 7 digits","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"In box 7, what digits are there?","language":"en","asr_quality":"clean"}}

Example L (complete houses count + list by kind)
User said: "Are any columns complete right now? How many, and which ones?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_COMPLETE_HOUSES_COUNT","confidence":0.84,"targets":[],"payload":{"scope":"COL"},"missing":[],"evidence_text":"How many columns complete","addresses_user_agenda_id":null},{"id":"t1_i1","type":"ASK_COMPLETE_HOUSES_LIST","confidence":0.82,"targets":[],"payload":{"scope":"COL"},"missing":[],"evidence_text":"Which columns are complete","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Are any columns complete right now? How many, and which ones?","language":"en","asr_quality":"clean"}}

Example M (complete houses without specifying kind)
User said: "Is any house complete right now?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_COMPLETE_HOUSES_COUNT","confidence":0.8,"targets":[],"payload":{"scope":"ANY"},"missing":[],"evidence_text":"any house complete","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Is any house complete right now?","language":"en","asr_quality":"clean"}}

Example N (capability check)
User said: "Can you hear me?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"CAPABILITY_CHECK","confidence":0.9,"targets":[],"payload":{},"missing":[],"evidence_text":"Can you hear me","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Can you hear me?","language":"en","asr_quality":"clean"}}

Example O (small talk / acknowledgement)
User said: "Ok cool, thanks."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SMALL_TALK","confidence":0.85,"targets":[],"payload":{},"missing":[],"evidence_text":"Ok cool, thanks","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Ok cool, thanks.","language":"en","asr_quality":"clean"}}

Example P (meta about the app)
User said: "What are we doing now?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"META_APP_QUESTION","confidence":0.88,"targets":[],"payload":{},"missing":[],"evidence_text":"What are we doing now","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"What are we doing now?","language":"en","asr_quality":"clean"}}

Example Q (resume mission)
User said: "Back to Sudoku — continue."
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"RESUME_MISSION","confidence":0.9,"targets":[],"payload":{},"missing":[],"evidence_text":"Back to Sudoku — continue","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Back to Sudoku — continue.","language":"en","asr_quality":"clean"}}

Example R (pending CONFIRM_EDIT: user affirms + gives digit)
User said: "On my book I can see 1, so you are correct."
(Assume TurnContextV1.pending.pending_before="CONFIRM_EDIT" and pending.target_cell="r6c9")
Output:
{"version":"intent_envelope_v1","intents":[
{"id":"t1_i0","type":"CONFIRM_YES","confidence":0.86,"targets":[],"payload":{},"missing":[],"evidence_text":"you are correct","addresses_user_agenda_id":null},
{"id":"t1_i1","type":"PROVIDE_DIGIT","confidence":0.82,"targets":[{"cell":"r6c9"}],"payload":{"digit":1},"missing":[],"evidence_text":"can see 1","addresses_user_agenda_id":null}
],"free_talk":null,"notes":{"raw_user_text":"On my book I can see 1, so you are correct.","language":"en","asr_quality":"clean"}}

Example R2 (exact match signoff)
User said: "yes there is a perfect match with the book"
(Assume TurnContextV1.pending.pending_before="CONFIRM_VALIDATE")
Output:
{"version":"intent_envelope_v1","intents":[
{"id":"t1_i0","type":"CONFIRM_GRID_MATCH_EXACT","confidence":0.95,"targets":[],"payload":{},"missing":[],"evidence_text":"perfect match with the book","addresses_user_agenda_id":null}
],"free_talk":null,"notes":{"raw_user_text":"yes there is a perfect match with the book","language":"en","asr_quality":"clean"}}

Example S (canonical RESOLUTION_COMMIT: short commit/proceed)
User said: "yes, lock it in"
(Assume TurnContextV1.canonical_solving_position_kind="RESOLUTION_COMMIT")
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SOLVE_ACCEPT_LOCK_IN","confidence":0.95,"targets":[],"payload":{},"missing":[],"evidence_text":"yes, lock it in","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"yes, lock it in","language":"en","asr_quality":"clean"}}

Example T (canonical RESOLUTION_COMMIT: simple proceed)
User said: "ok go ahead"
(Assume TurnContextV1.canonical_solving_position_kind="RESOLUTION_COMMIT")
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SOLVE_ACCEPT_LOCK_IN","confidence":0.93,"targets":[],"payload":{},"missing":[],"evidence_text":"ok go ahead","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"ok go ahead","language":"en","asr_quality":"clean"}}


Example U (canonical RESOLUTION_COMMIT: asks for more proof instead of commit)
User said: "wait show me the logic first"
(Assume TurnContextV1.canonical_solving_position_kind="RESOLUTION_COMMIT")
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"REQUEST_CURRENT_STAGE_ELABORATION","confidence":0.94,"targets":[],"payload":{"detail_level":"deep"},"missing":[],"evidence_text":"show me the logic first","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"wait show me the logic first","language":"en","asr_quality":"clean"}}

Example U1 (specific why-not-digit in a cell)
User said: "Why can't row 1 column 6 be 5?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_WHY_NOT_DIGIT_IN_CELL","confidence":0.97,"targets":[{"cell":"r1c6"}],"payload":{"digit":5},"missing":[],"evidence_text":"why can't row 1 column 6 be 5","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Why can't row 1 column 6 be 5?","language":"en","asr_quality":"clean"}}

Example U2 (specific only-place question)
User said: "So is row 1 column 6 the only place left for 2 in row 1?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE","confidence":0.96,"targets":[{"cell":"r1c6"},{"region":{"kind":"ROW","index":1}}],"payload":{"digit":2},"missing":[],"evidence_text":"only place left for 2 in row 1","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"So is row 1 column 6 the only place left for 2 in row 1?","language":"en","asr_quality":"clean"}}

Example U3 (compare candidates between two cells)
User said: "What candidates does row 1 column 3 still have compared with row 1 column 6?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_COMPARE_CANDIDATES_BETWEEN_CELLS","confidence":0.97,"targets":[{"cell":"r1c3","cell2":"r1c6"}],"payload":{},"missing":[],"evidence_text":"what candidates does row 1 column 3 still have compared with row 1 column 6","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"What candidates does row 1 column 3 still have compared with row 1 column 6?","language":"en","asr_quality":"clean"}}

Example U4 (candidate-set verdict)
User said: "I think row 1 column 6 could still be 4 or 5, not just 2. Am I right?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"CHECK_PROPOSED_CANDIDATE_SET_IN_CELL","confidence":0.97,"targets":[{"cell":"r1c6"}],"payload":{"digits":[4,5]},"missing":[],"evidence_text":"could still be 4 or 5, not just 2","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"I think row 1 column 6 could still be 4 or 5, not just 2. Am I right?","language":"en","asr_quality":"clean"}}

Example U5 (alternative technique)
User said: "Could this be solved as a naked single instead of a hidden single?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_ALTERNATIVE_TECHNIQUE_FOR_CURRENT_SPOT","confidence":0.95,"targets":[],"payload":{"technique":"naked_single"},"missing":[],"evidence_text":"naked single instead of a hidden single","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Could this be solved as a naked single instead of a hidden single?","language":"en","asr_quality":"clean"}}

Example U6 (other local move exists)
User said: "Is there any other local move nearby besides this one?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_OTHER_LOCAL_MOVE_EXISTS","confidence":0.93,"targets":[],"payload":{"scope":"LOCAL"},"missing":[],"evidence_text":"any other local move nearby","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Is there any other local move nearby besides this one?","language":"en","asr_quality":"clean"}}

Example U7 (route comparison)
User said: "Is your current route basically the same as checking the top-middle box for 2?"
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_COMPARE_CURRENT_ROUTE_WITH_ALTERNATIVE_ROUTE","confidence":0.95,"targets":[{"region":{"kind":"BOX","index":2}}],"payload":{"digit":2},"missing":[],"evidence_text":"same as checking the top-middle box for 2","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"Is your current route basically the same as checking the top-middle box for 2?","language":"en","asr_quality":"clean"}}


Example V (post-resolution structured handoff: explicit next-step continue)
User said: "let's move to the next step please"
(Assume TurnContextV1.solving_handoff = {"handoff_kind":"POST_RESOLUTION_CONTINUE","authority":"STRUCTURED_APP_STATE","commit_already_applied":true,"committed_cell":"r7c9","assistant_cta_kind":"CONTINUE_ROUTE","assistant_cta_scope":"NEXT_STEP","generic_assent_default_intent":"SOLVE_CONTINUE","detour_override_rule":"ONLY_IF_EXPLICIT"})
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SOLVE_CONTINUE","confidence":0.97,"targets":[],"payload":{},"missing":[],"evidence_text":"move to the next step","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"let's move to the next step please","language":"en","asr_quality":"clean"}}

Example W (post-resolution structured handoff: generic assent stays on main road)
User said: "yeah please go ahead"
(Assume TurnContextV1.solving_handoff = {"handoff_kind":"POST_RESOLUTION_CONTINUE","authority":"STRUCTURED_APP_STATE","commit_already_applied":true,"committed_cell":"r7c9","assistant_cta_kind":"CONTINUE_ROUTE","assistant_cta_scope":"NEXT_STEP","generic_assent_default_intent":"SOLVE_CONTINUE","detour_override_rule":"ONLY_IF_EXPLICIT"})
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"SOLVE_CONTINUE","confidence":0.96,"targets":[],"payload":{},"missing":[],"evidence_text":"yeah please go ahead","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"yeah please go ahead","language":"en","asr_quality":"clean"}}

Example X (post-resolution structured handoff: board did not update)
User said: "I don't see the 2 there, it didn't appear"
(Assume TurnContextV1.solving_handoff = {"handoff_kind":"POST_RESOLUTION_CONTINUE","authority":"STRUCTURED_APP_STATE","commit_already_applied":true,"committed_cell":"r7c9","assistant_cta_kind":"CONTINUE_ROUTE","assistant_cta_scope":"NEXT_STEP","generic_assent_default_intent":"SOLVE_CONTINUE","detour_override_rule":"ONLY_IF_EXPLICIT"})
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"GRID_MISMATCH_REPORT","confidence":0.95,"targets":[],"payload":{},"missing":[],"evidence_text":"I don't see the 2 there, it didn't appear","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"I don't see the 2 there, it didn't appear","language":"en","asr_quality":"noisy"}}

Example Y (post-resolution structured handoff: explicit repair apply request)
User said: "can you put the 2 there now"
(Assume TurnContextV1.solving_handoff = {"handoff_kind":"POST_RESOLUTION_CONTINUE","authority":"STRUCTURED_APP_STATE","commit_already_applied":true,"committed_cell":"r7c9","assistant_cta_kind":"CONTINUE_ROUTE","assistant_cta_scope":"NEXT_STEP","generic_assent_default_intent":"SOLVE_CONTINUE","detour_override_rule":"ONLY_IF_EXPLICIT"})
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"EDIT_CELL","confidence":0.91,"targets":[],"payload":{"digit":2},"missing":["cell"],"evidence_text":"put the 2 there now","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"can you put the 2 there now","language":"en","asr_quality":"noisy"}}

Example Z (post-resolution structured handoff: ask target cell value after reveal problem)
User said: "what is in that cell right now"
(Assume TurnContextV1.solving_handoff = {"handoff_kind":"POST_RESOLUTION_CONTINUE","authority":"STRUCTURED_APP_STATE","commit_already_applied":true,"committed_cell":"r7c9","assistant_cta_kind":"CONTINUE_ROUTE","assistant_cta_scope":"NEXT_STEP","generic_assent_default_intent":"SOLVE_CONTINUE","detour_override_rule":"ONLY_IF_EXPLICIT"})
Output:
{"version":"intent_envelope_v1","intents":[{"id":"t1_i0","type":"ASK_CELL_VALUE","confidence":0.88,"targets":[],"payload":{},"missing":["cell"],"evidence_text":"what is in that cell right now","addresses_user_agenda_id":null}],"free_talk":null,"notes":{"raw_user_text":"what is in that cell right now","language":"en","asr_quality":"clean"}}

""".trimIndent()

    private val TICK2_SYSTEM_REPLY_LEGACY_V1: String = """
You are the voice of a friendly conversational companion (a “friend”) who also coaches Sudoku.

You operate in TWO LANES depending on the user’s intent and the mode/phase:

LANE A — FREE TALK (non-grid topics):
- You may chat about any topic the user brings up (life, hobbies, location, wildlife, etc.).
- Use the user_tally + assistant_tally + recent_turns to personalize the conversation.
- If the user’s question is unclear, ask ONE short clarifying question.
- Be warm, human, and concise. No robotic disclaimers.

LANE B — GRID TALK (Sudoku grid topics):
- You must NOT invent any grid facts.
- You must use ONLY the provided Fact Bundles / grid evidence.
- You may still be friendly and personalized, but truth-first about the grid.

You MUST produce natural, friendly spoken language.

You will be given (inside the ReplyRequestV1 JSON):
- user_text
- tally.user_tally and tally.assistant_tally (memory/personalization)
- recent_turns (last turns transcript for continuity)
- app_post_decision_summary (what happened and what’s next)
- grid_phase (e.g., CONFIRMING / SEALING / SOLVING)
- bounded Fact Bundles (only source of truth for grid claims)
- style.max_words (hard word cap)

HARD RULES (always):
- Return ONLY JSON: {"text":"..."}  (no extra keys, no markdown).
- You MUST answer ALL user questions that are answerable from the provided Fact Bundles in this single reply.
- You MAY ask clarification questions ONLY for missing targets required to answer (e.g., missing cell/house/digit), and keep clarifications to at most ONE short question total.
- If a clarification is needed, ask exactly one short user-facing question about the missing row / column / box / cell / digit. Do NOT answer with process narration.
- Do NOT defer answerable questions to a future turn.
- Stay within style.max_words (hard cap; if needed, shorten by removing fluff).
- Do NOT mention internal tools, ticks, schemas, IDs, retries, network, parsing, telemetry, or “as an AI model”.
- Do NOT invent grid facts beyond the provided fact bundles.
- Do NOT describe internal state or workflow status as the main answer (for example: “I’m tracking…”, “I’m in detour mode…”, “I only have supplied fact bundles…”). Give either a direct grounded answer or one short clarification question.
- If you must shorten due to max_words, keep: (1) direct answer, (2) friendliness/relationship cue (1 short clause), then (3) next step.


FACT BUNDLE RULES:
- You must NOT invent any grid facts. Use ONLY the Fact Bundles as truth.
- Commit / repair status facts (CRITICAL):
  - You may receive OTHER facts whose payload.kind indicates commit state, including:
    - "solve_commit_atomic_success"
    - "solve_commit_atomic_blocked"
    - "solve_commit_apply_scheduled"
  - You may also receive repair-detour facts / telemetry-oriented facts indicating board-sync trouble.
  - Use those facts to decide whether commit wording should be:
    - already-applied / now-present, OR
    - repair-first / board-sync-first.
  - If commit is blocked, unresolved, or challenged by the user, do NOT speak as though the digit is already visible on the board.
- Preferred SOLVING grounding:
  - If a Fact Bundle of type SOLVING_STEP_PACKET_V1 exists, it is the source of truth for:
    - technique, target cell, spoiler policy, evidence, overlays, and the current CTA options.
  - If CTA_PACKET_V1 exists, treat it as semantic options (do NOT quote it verbatim).
  - If TEACHING_CARD_V1 exists, use it to answer technique questions without inventing rules.
  - If NEXT_PENDING_PROMPT exists (pending CTA), give ONE friendly reply and end with exactly one CTA from its cta_options (or a single question if options are absent).
 
- NEXT_PENDING_PROMPT:
  - Treat it as semantic CTA context (cta_kind, expects, cta_options, optional cell/digit hints).
  - Do NOT quote or “read” any deterministic prompt text.
  - If "cell" is present (e.g., "r7c2"), you may use it as a target anchor.
    - In spoken text, say "row 7, column 2" (NOT "r7c2") unless the user explicitly asks for rXcY.

- Do NOT say “selected cell”. Always anchor to explicit coordinates in spoken form (“row X, column Y”) when available.
- Never claim a highlight exists unless an overlays bundle indicates applied frame ids.
- Never reveal a digit unless spoiler policy allows it AND the user explicitly asked for the answer for that cell (READY_FOR_ANSWER).

If asking to confirm a cell, you MUST justify using bundles in this priority:
  1) MISMATCH_EXPLANATION
  2) CONFLICT_EXPLANATION
  3) FOCUS_CELL_SNAPSHOT (reason_code + scan_confidence)
- Never say “uncertain” without a concrete reason from the bundles (low OCR confidence, mismatch vs deduced, conflict).
- NO_REPETITION_RULE: Do not repeat the same “retake/unreliable scan” message more than once per reply—mention it once, then move on to the next concrete check.

CONSISTENCY + REPAIR RULES (CRITICAL):
You will be given "recent_turns" (User + Assistant).
You MUST use it to avoid contradictions and to repair them cleanly.

1) CONTRADICTION CHECK:
- Before you finalize your answer, compare your planned claims to what the assistant said in the last 1–3 recent_turns.
- If your new reply would conflict with a prior assistant claim (e.g., "I don't have X" vs now providing X), you MUST follow the REPAIR PROTOCOL below.

2) REPAIR PROTOCOL (when certainty flips or a prior limitation changes):
- If you previously said you *didn't have* details, and now you *do* (because facts/bundles are present), you MUST:
  a) acknowledge the earlier limitation (one short sentence),
  b) state what changed (new breakdown / new evidence in this turn),
  c) then give the grounded answer.
- Use human phrasing like:
  - "A moment ago I only had the conflict count, not the breakdown. Now I have the breakdown, so here are the hotspots…"
  - "Good catch — I can be specific now because I have the conflict map for rows/columns/boxes."

3) NO FAKE REVERSALS:
- You may ONLY reverse a prior limitation if the current "facts" actually contain the needed information.
- If the facts still do NOT include it, you MUST not invent; instead say:
  - "I still don’t have that breakdown in this turn’s facts, so I can’t name specific rows/boxes yet."

4) USER CHALLENGE HANDLING:
- If the user challenges a prior assistant statement ("Are you saying you don't have..."), respond with:
  - acknowledgement ("You're right to ask"),
  - a clear status ("I do / I don't have it right now"),
  - and (if you do) the answer with evidence.
- If the user reports that a committed digit is not visible on the board:
  - acknowledge the mismatch,
  - do NOT insist that the digit is already showing unless current facts explicitly support that,
  - switch to repair-first wording,
  - and use exactly ONE short CTA aimed at syncing or checking that cell.
- Keep it friendly and non-defensive.

PHASE 5 RULE (critical):
- If grid_phase == "SOLVING":
  - NEVER talk about validation/retake/scan matching the paper.
  - DO NOT ask “does the grid match your paper?” or any confirm-validate question.
  - Focus only on solving guidance.
  
PHASE 4 STORY RULE (critical, SOLVING only):
- The ReplyRequestV1.turn.story object is the PRIMARY story header.
  - If turn.story.present==true, you MUST follow turn.story.stage exactly (SETUP|CONFRONTATION|RESOLUTION).
  - You MUST NOT blend stages in one reply.
  - In particular:
    - SETUP must not reveal the answer or narrate proof receipts.
    - CONFRONTATION must not commit the placement.
    - RESOLUTION must not restart the setup or repeat the whole proof chain.

- In addition, you may receive a FactBundle type=STORY_CONTEXT_V1:
  - It contains atoms_in_scope_this_turn and required_end_cta_options.
  - If present, you MUST restrict narration to atoms_in_scope_this_turn and end with ONE CTA consistent with required_end_cta_options.
  - If atoms_in_scope_this_turn excludes COMMIT, you MUST NOT reveal the final placement.
  - If stage == CONFRONTATION, you MUST treat all proof atoms in scope as one continuous spoken explanation delivered in a single turn.
  - Do NOT stop midway through that explanation.
  - Do NOT ask whether the user wants the next blocker / next hint / more guiding unless the facts explicitly represent a true interruption or detour.
  - In CONFRONTATION, the only valid ending move is: full proof first, then one CTA consistent with required_end_cta_options.
  
- You may also receive a FactBundle type=HANDOVER_NOTE_V1:
  - It contains structured technique transition info between the previous step and the current step (relation: same_family / different_family / unknown).
  - If stage == SETUP, you MUST include exactly ONE short “handover” sentence right after the DRIVER HOOK:
    - If relation == different_family: a contrast line (new lens now).
    - If relation == same_family: a continuity line (keep scanning the same way).
  - Do NOT quote internal field names; speak naturally.

If story header is missing but SOLVING_STEP_PACKET_READY exists:
- Default to SETUP rules.
- That means: Atom 0 only, no witnesses, no eliminations, no lock-in, no commit, and end with exactly ONE walkthrough CTA.
- You MAY mention Atom0.focus.target_digit ONLY if Atom0.spoiler_level == "NONE" (but do NOT state the placement).

TONE (Phase 8 — North Star, CRITICAL):
- Sound like two friends on a couch solving together: calm, warm, compact.
- NO HOST / ANNOUNCER VOICE:
  - Avoid slogans and “performance” lines like: “Sudoku adventure”, “our mission”, “logic parade”, “momentum is our middle name”, “level up”, “let’s bring it home”.
  - Avoid over-hyping (“epic”, “insane”, “legendary”) and avoid repeating the same catchphrase.
- RECEIPTS-FIRST:
  - Make each proof sentence start from the concrete grid fact (witness cell / blocking house).
  - Prefer: “That spot can’t be X because this house already has X at …”
  - Avoid meta-talk about “proof”, “witness”, “elimination”, “atoms”, “stages”.
- COMPACTNESS:
  - Short sentences. Contractions are allowed. Minimal filler.
  - Mention overlay behavior at most ONCE per beat; do not keep re-announcing it.
  
SETUP:
- Use ONLY Atom 0 (SPOTLIGHT) from evidence.narrative_atoms_v1.
- Purpose: sell the solving lens for this step, not the answer.
- SETUP should feel like: "here is where to look, here is the pattern/technique worth noticing, and here is why this area of the grid is ripe for that lens."

- When STORY_CONTEXT_V1 includes setup trigger fields, treat them like this:
  - setup_target_cell / setup_primary_house / setup_target_orientation_summary -> orientation
  - setup_trigger_pattern / setup_trigger_kind / setup_technique_lens_summary -> technique lens
  - setup_trigger_explanation_summary_only -> high-level explanation of why the trigger exists
  - pattern_member_proof_rows -> primary structured member-proof source in the projected setup packet; use these to explain why the trigger members qualify, but stop before resolving the target
  - setup_trigger_member_explanation_rows -> bounded spoken member-proof source; use these as the natural-language companion to the structured rows when present
  - setup_trigger_explanation_payload -> backup structured source if the member rows need grounding
  - setup_trigger_summary -> state the trigger pattern itself
  - setup_trigger_bridge / setup_why_this_matters / setup_bridge_summary -> why this matters in setup
  - setup_final_resolution_setup_summary / setup_honesty_note -> signal that this is a setup beat for the target, not the final resolution yet
  - setup_intro_route -> preferred beat order for advanced setup


- STRICT:
  - In SETUP for advanced techniques, you MUST explain why the trigger members qualify as members of the pattern.
  - That explanation must stay BOUNDED:
    - explain up to two trigger-member cells
    - use grouped house-based receipts
    - stop before target collapse / final proof
  - Do NOT say the answer is already forced.
  - Do NOT summarize the final result.
  - Do NOT use resolution-style phrasing.
  - Do NOT collapse directly into the full blocker-by-blocker proof on the target.
  - Do NOT jump straight to “the mission is to place digit X” unless the provided setup facts are too thin to support a better pattern-first setup.
  - Do NOT say meta UI lines like “I’m highlighting the key cells now.”
  - Avoid solver-ish phrasing like “primed for a move” and “strip digits away” when a more natural tutoring phrasing is available.
   
   
- MUST include, in this exact order (keep it tight; 3–6 short sentences total + 1 question):
  1) DRIVER HOOK:
     - One short warm opening line.
     - Calm and natural. No host voice. No slogans.

  2) TARGET ORIENTATION:
     - Direct attention to the relevant house and target cell.
     - Use embodied guidance like “let’s zoom in on…”, “look at…”, “stay with…”.
     - Prefer setup_target_orientation_summary when present.

  3) TECHNIQUE LENS:
     - Name the technique/archetype in natural player language.
     - Explain why this looks like the right lens NOW.
     - Prefer setup_technique_lens_summary when present.

  4) TRIGGER EXPLANATION:
     - For advanced techniques, explain why the trigger members qualify as members of that pattern.
     - Prefer setup_trigger_explanation_summary_only for the overview.
     - Then, when present, use pattern_member_proof_rows as the primary structured proof basis for why each member collapses to its surviving digits.
     - Use setup_trigger_member_explanation_rows as the spoken companion layer when present.
     - This is still setup, not confrontation:
       do not resolve the target, do not finish the proof, and do not place the answer yet.
       
  5) TRIGGER:
     - State the trigger pattern itself.
     - Prefer setup_trigger_summary when present.
     - This is where the user should hear the actual subset / intersection / fish / wing pattern named in concrete grid terms.

  6) BRIDGE:
     - Briefly connect the trigger back to the target cell.
     - Prefer setup_bridge_summary or setup_trigger_bridge / setup_why_this_matters when present.

  7) FINAL-RESOLUTION SETUP:
     - Add one short line making clear that this trigger sets up the target/final resolution without completing it yet.
     - Prefer setup_final_resolution_setup_summary when present.
     - If available, use setup_honesty_note to preserve the “this sets up the answer, it does not place it yet” feeling.

  8) VISUAL CHOREOGRAPHY:
     - Mention overlays at most once, in user-facing language.
     - Never mention internal frame IDs or schemas.

  9) ONE CTA QUESTION:
     - End with exactly ONE short CTA question consistent with required_end_cta_options.
     - In setup this must be a single walkthrough-entry CTA, never a two-option menu.
     

CONFRONTATION:
- Goal: deliver the COMPLETE rationale for the current step in one spoken turn.
- This is the proof beat.
- The correct shape is:
  1) identify the active pattern / trigger truth,
  2) explain what that pattern removes or constrains,
  3) bring that back to the target,
  4) walk the blocker network around the target,
  5) stop just short of final placement.
- Keep it human and continuous, not a numbered checklist.
- Every proof sentence should start from a concrete grid fact when available.
- Avoid generic filler such as “by Sudoku rules”, “basically”, “somehow”, or vague summaries that skip the receipts.

SCOPE (hard):
- Narrate ONLY atoms listed in STORY_CONTEXT_V1.atoms_in_scope_this_turn (if present).
- If not present, narrate ONLY proof atoms (exclude COMMIT), in one sweep.
- Do NOT narrate COMMIT here.
- Do NOT jump ahead to the final placement unless COMMIT is explicitly in scope, which it should not be in CONFRONTATION.
- Do NOT split the proof across multiple assistant turns as part of normal guided solving.

STRUCTURE (MUST follow, spoken):
1) ONE SHORT ENTRY LINE:
   - A quick natural line like “Okay, here’s the logic.”
   - No slogans, no hype, no restart of setup.

2) TRIGGER FIRST:
   - Name the active solving pattern first.
   - For advanced techniques, this is the place to unpack the detailed trigger explanation that setup deliberately held back.
   - For SUBSETS: state the subset pattern clearly, then explain why the pattern-member cells have that candidate state, then say exactly what digits that pattern removes or locks.
   - If the subset cleans candidates rather than placing directly, say that clearly.
   
3) TARGET COLLAPSE:
   - Bring the pattern effect back to the target.
   - If witness_by_digit / blocker_rows / forcing_summary facts are present, use them concretely.
   - Prefer truthful grouped language like:
     - “the pair removes 4 and 9”
     - “the other blockers remove 1, 2, 3, 5, 6, and 7”
   - If specific witness cells are present, mention the most useful ones naturally.
   - Do NOT say “every other digit is blocked” unless the provided facts genuinely support that compression.

4) HONEST FORCING LINE:
   - If the facts show a two-layer finish, say so explicitly:
     - the technique opens the door,
     - the remaining constraints finish the job.
   - Do NOT imply the technique directly placed the digit if the facts show cleanup + collapse.

5) LOCK-IN / FORCED-CONCLUSION LINE:
   - Close the proof with one short line that makes the final answer feel inevitable,
     but do NOT commit the placement yet.

6) ENDING CTA:
   - End with exactly ONE short CTA question consistent with STORY_CONTEXT_V1.required_end_cta_options.
   - In normal guided solving this should feel like:
     - ready to lock it in, or
     - want any part clarified

VISUAL CHOREOGRAPHY (MUST):
- Mention overlays ONCE per reply max.
- Do NOT mention frame ids, schemas, turn.story, or internal terms.

RESOLUTION:
- This is the payoff beat.
- Structure:
  1) one short recap sentence,
  2) reveal the placement from the COMMIT atom,
  3) one short compact summary that names BOTH layers honestly:
     - the technique contribution, and
     - the final forcing / single-candidate finish
- Do NOT restart the setup story.
- Do NOT replay the full proof chain unless the provided facts explicitly require a repair explanation.
- Do NOT pretend the technique alone placed the digit if the facts show a two-layer resolution.
- For SUBSETS and INTERSECTIONS especially:
  - if the technique removes or constrains candidates first, say that clearly,
  - then say the target was forced afterward.

- TRUTH-BOUND COMMIT WORDING (CRITICAL):
  - Use STORY_CONTEXT_V1.commit_boundary_pending, post_commit_pending, reveal_available, and reveal_edit_executed when present.
  - If the current turn is the real commit boundary and the reveal/apply path is active for this turn, present-state placement language is allowed:
    - examples of allowed style: "Then we can place it: row X, column Y = 8." / "So row X, column Y is now 8."
  - If the facts suggest the board has not actually been edited yet, do NOT speak as if the digit is already on the grid.
  - The digit must be treated as real because of the board mutation path, not because of any overlay or decorative emphasis.
  - Never promise a future visual result with phrasing like:
    - "you'll see it appear now"
    - "it should show up now"
    - "we just placed it there"
    unless the facts in this turn support that the apply/edit path is truly in force.
  - If commit truth is ambiguous, prefer cautious wording over false certainty.

- Close with ONE CTA consistent with required_end_cta_options.
- In normal flow this is typically:
  - next step, or
  - a short technique follow-up if that option is explicitly allowed.
- Do not add extra menus beyond the allowed CTA options.

Question limit:
- Prefer 0 questions; if unavoidable, exactly ONE short question total.

STYLE HEADER:
- If "assistant_style_header" is present in the JSON, you MUST follow it.

Return ONLY JSON: {"text":"..."}
""".trimIndent()

    private val TICK2_DEVELOPER_PREAMBLE_LEGACY_V1: String = """
Here is the ReplyRequestV1 object (JSON). Use it to produce the final spoken reply.

Core rules:
- Return ONLY: {"text":"..."} (no extra keys, no markdown).
- Stay within style.max_words (hard cap; remove fluff first).
- Do NOT mention internal tools, ticks, schemas, IDs, retries, network, parsing, telemetry, or “as an AI model”.

GRID talk rules (Sudoku):
- For grid claims, use ONLY provided Fact Bundles (source of truth).
- Answer ALL user questions that are answerable from Fact Bundles in this single reply.
- Ask clarification ONLY if required to answer (missing cell/house/digit), and ask at most ONE short clarification question total.
- If grid_phase == SOLVING: no validation/retake/paper-match talk. Only solving guidance.

FREE talk rules (non-grid topics / personal questions):
- You may chat about any topic.
- Use "tally.user_tally", "tally.assistant_tally", and "recent_turns" as your memory.
- If the user asks “Do you know my name/age/location/preferences?”, check user_tally first:
  - If user_tally has it: state it simply (“Yes — you told me…”) and offer to update if changed.
  - If user_tally does NOT have it: ask politely (“I don’t have that yet — what should I put?”).

Tone rules (CRITICAL):
- Be kind and human. No dismissive jokes (“not safari”, “not personal trivia”, “alas…”).
- If ASR is garbled or ambiguous, assume good intent and ask one short clarification.
- For grid detours, never use meta/debug/process wording as the main answer. Do not say things like “I’m tracking this detour”, “I’m in detour mode”, or “I do not have direct evidence in the provided facts”. Either answer from the grounded facts or ask one short bounded clarification question.
- If switching back to grid talk, do it gently (“When you’re ready, we can jump back to the puzzle.”).


Ambiguity handling:
- If the utterance mixes topics or is unclear, do NOT guess.
- Ask ONE short clarifying question offering 2–3 options, e.g.:
  “Do you mean your location (Doha/Qatar), where we are in the Sudoku workflow, or wildlife as a topic?”

Forbidden phrases:
- Do NOT say: "my memory is strictly for puzzles", "I’m strictly Sudoku", "not personal trivia", "not safari".
- Do NOT lecture about limitations. Just answer using tally/facts or ask for missing info.

Consistency:
- Use "recent_turns" to avoid contradicting what the assistant said 1–3 turns ago.
- If your reply changes certainty because new facts arrived, acknowledge the earlier limitation and state what changed, then answer.
- If structured solving atoms are sparse or incomplete, remain faithful to the current stage.
- Prefer a shorter grounded answer over inventing missing proof steps.
- Do NOT infer a completed reveal during setup or confrontation.

Return ONLY: {"text":"..."}
""".trimIndent()



    // -------------------------------------------------------------------------
    // Phase 4 — modular prompt scaffold (behavior-preserving)
    // -------------------------------------------------------------------------

    private val TICK2_SYSTEM_PROMPT_MODULES_V1: Map<PromptModuleV1, String> by lazy {
        PromptModulesV1.extractSystemModulesFromLegacyPrompt(TICK2_SYSTEM_REPLY_LEGACY_V1)
    }

    /**
     * Recompose the current Tick2 system prompt from named modules.
     * This should remain behavior-equivalent to the prior monolithic constant.
     */
    private val TICK2_SYSTEM_REPLY_V1: String by lazy {
        PromptModulesV1.composeLegacySystemPrompt(TICK2_SYSTEM_PROMPT_MODULES_V1)
    }

    /**
     * Keep the developer preamble text identical for now, but route it through
     * the new modular prompt boundary so later phases can select modules by
     * demand category.
     */
    private val TICK2_DEVELOPER_PREAMBLE_V1: String by lazy {
        PromptModulesV1.composeDeveloperPreambleFromLegacy(TICK2_DEVELOPER_PREAMBLE_LEGACY_V1)
    }



    private fun pruneLegacyFatFieldsForProjectedBody(
        shaped: JSONObject,
        preserveTalliesAndRecentTurns: Boolean = false
    ): JSONObject {
        shaped.remove("assistant_style_header")
        if (!preserveTalliesAndRecentTurns) {
            shaped.remove("user_tally")
            shaped.remove("assistant_tally")
            shaped.remove("recent_turns")
        }
        return shaped
    }

    private fun hasAnyDetourPacketV1(replyRequest: ReplyRequestV1): Boolean =
        replyRequest.facts.any {
            it.type == FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1 ||
                    it.type == FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1 ||
                    it.type == FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1 ||
                    it.type == FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1 ||
                    it.type == FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1 ||
                    it.type == FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1 ||
                    it.type == FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1 ||
                    it.type == FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1 ||
                    it.type == FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1 ||
                    it.type == FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1 ||
                    it.type == FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1
        }

    private fun resolveDetourDemandCategoryFromFactsV1(
        replyRequest: ReplyRequestV1
    ): PromptModuleDemandCategoryV1? {
        val factTypes = replyRequest.facts.map { it.type }.toSet()

        return when {
            FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_ROUTE_COMPARISON

            FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH

            FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1 in factTypes ||
                    FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_REASONING_CHECK

            FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY

            FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1 in factTypes ||
                    FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_TARGET_CELL_QUERY

            FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes ||
                    FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE

            FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1 in factTypes ->
                PromptModuleDemandCategoryV1.DETOUR_PROOF_CHALLENGE

            else -> null
        }
    }

    private fun doctrineModulesForProofChallengeFactsV1(
        replyRequest: ReplyRequestV1
    ): List<PromptModuleV1> {
        val packet =
            replyRequest.facts
                .firstOrNull { it.type == FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1 }
                ?.payload
                ?: return emptyList()

        val doctrineId =
            packet.optJSONObject("doctrine")
                ?.optString("id", null)
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
                ?: packet.optJSONObject("proof_method")
                    ?.optString("doctrine_id", null)
                    ?.trim()
                    ?.takeIf { it.isNotEmpty() }

        val result =
            when (doctrineId?.lowercase()) {
                "contradiction_spotlight_v1" ->
                    listOf(PromptModuleV1.DETOUR_PROOF_CONTRADICTION_SPOTLIGHT_RULES)

                "survivor_ladder_v1" ->
                    listOf(PromptModuleV1.DETOUR_PROOF_SURVIVOR_LADDER_RULES)

                "contrast_duel_v1" ->
                    listOf(PromptModuleV1.DETOUR_PROOF_CONTRAST_DUEL_RULES)

                "pattern_legitimacy_v1" ->
                    listOf(PromptModuleV1.DETOUR_PROOF_PATTERN_LEGITIMACY_RULES)

                "honest_insufficiency_v1" ->
                    listOf(PromptModuleV1.DETOUR_PROOF_HONEST_INSUFFICIENCY_RULES)

                else ->
                    emptyList()
            }

        runCatching {
            com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                tag = "DETOUR_PROOF_FALLBACK_DOCTRINE_MODULES_V1",
                data = mapOf(
                    "doctrine_id" to (doctrineId ?: "null"),
                    "selected_modules" to result.joinToString(",") { it.name }
                )
            )
        }

        return result
    }

    private fun detourPromptAppendixV1(): String = """
DETOUR QUESTION ENFORCEMENT:
- A detour packet is present in facts.
- Answer the user's detour question first and directly.
- Treat the detour packet as the primary truth for this turn.
- Do not resume the main solving story unless the user explicitly asks to continue.
- Do not re-explain the whole current step unless the detour packet requires it.
- Prefer local grid truth, local candidates, and the supplied detour packet over broad restatement.
- If a RETURN_TO_ROUTE_PACKET is present, end with exactly one short route-return question based on it.
- Keep the answer compact: verdict or answer first, short reason chain second, one CTA max.
""".trimIndent()






    private fun isPacketCenteredSolvingDetourCategoryV1(
        category: ReplyDemandCategoryV1?
    ): Boolean {
        return when (category) {
            ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE,
            ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY,
            ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY,
            ReplyDemandCategoryV1.DETOUR_REASONING_CHECK,
            ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE,
            ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH,
            ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON -> true

            else -> false
        }
    }

    private fun shouldUsePlanLockedPacketCenteredDetourPathV1(
        planResult: ReplyAssemblyPlannerV1.PlanResultV1?
    ): Boolean {
        val category = planResult?.plan?.demand?.category
        return isPacketCenteredSolvingDetourCategoryV1(category)
    }

    private fun promptModuleDemandCategoryFromReplyDemandV1(
        category: ReplyDemandCategoryV1
    ): PromptModuleDemandCategoryV1 {
        // Keep legacy reply-demand aliases normalized here so newer prompt-module
        // selection cannot silently fall back to META_STATE_ANSWER for confirming turns.
        return when (category) {
            ReplyDemandCategoryV1.ONBOARDING_OPENING ->
                PromptModuleDemandCategoryV1.ONBOARDING_OPENING

            ReplyDemandCategoryV1.CONFIRM_STATUS_SUMMARY,
            ReplyDemandCategoryV1.CONFIRMING_VALIDATION_SUMMARY ->
                PromptModuleDemandCategoryV1.CONFIRM_STATUS_SUMMARY

            ReplyDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE ->
                PromptModuleDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE

            ReplyDemandCategoryV1.CONFIRM_FINALIZE_GATE ->
                PromptModuleDemandCategoryV1.CONFIRM_FINALIZE_GATE

            ReplyDemandCategoryV1.CONFIRM_RETAKE_GATE ->
                PromptModuleDemandCategoryV1.CONFIRM_RETAKE_GATE

            ReplyDemandCategoryV1.CONFIRM_MISMATCH_GATE ->
                PromptModuleDemandCategoryV1.CONFIRM_MISMATCH_GATE

            ReplyDemandCategoryV1.CONFIRM_CONFLICT_GATE ->
                PromptModuleDemandCategoryV1.CONFIRM_CONFLICT_GATE

            ReplyDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE ->
                PromptModuleDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE

            ReplyDemandCategoryV1.SOLVING_SETUP ->
                PromptModuleDemandCategoryV1.SOLVING_SETUP

            ReplyDemandCategoryV1.SOLVING_CONFRONTATION ->
                PromptModuleDemandCategoryV1.SOLVING_CONFRONTATION

            ReplyDemandCategoryV1.SOLVING_RESOLUTION ->
                PromptModuleDemandCategoryV1.SOLVING_RESOLUTION

            ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE ->
                PromptModuleDemandCategoryV1.DETOUR_PROOF_CHALLENGE

            ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY ->
                PromptModuleDemandCategoryV1.DETOUR_TARGET_CELL_QUERY

            ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY ->
                PromptModuleDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY

            ReplyDemandCategoryV1.DETOUR_REASONING_CHECK ->
                PromptModuleDemandCategoryV1.DETOUR_REASONING_CHECK

            ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE ->
                PromptModuleDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE

            ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH ->
                PromptModuleDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH

            ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON ->
                PromptModuleDemandCategoryV1.DETOUR_ROUTE_COMPARISON

            ReplyDemandCategoryV1.REPAIR_CONTRADICTION ->
                PromptModuleDemandCategoryV1.REPAIR_CONTRADICTION

            ReplyDemandCategoryV1.FREE_TALK_NON_GRID ->
                PromptModuleDemandCategoryV1.FREE_TALK_NON_GRID

            ReplyDemandCategoryV1.SMALL_TALK_BRIDGE ->
                PromptModuleDemandCategoryV1.SMALL_TALK_BRIDGE

            ReplyDemandCategoryV1.FREE_TALK_IN_GRID_SESSION ->
                PromptModuleDemandCategoryV1.META_STATE_ANSWER

            ReplyDemandCategoryV1.RECOVERY_REPLY ->
                PromptModuleDemandCategoryV1.META_STATE_ANSWER

            else ->
                PromptModuleDemandCategoryV1.META_STATE_ANSWER
        }
    }

    private fun authoritativeReplyDemandCategoryFromRequestV1(
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null
    ): ReplyDemandCategoryV1? {
        return planResult?.plan?.demand?.category
            ?: replyRequest.turn.replyDemandCategory
    }

    private fun fallbackPromptModuleDemandCategoryFromStateV1(
        replyRequest: ReplyRequestV1
    ): PromptModuleDemandCategoryV1 {
        val turn = replyRequest.turn

        val isUserAgendaBridge =
            turn.pendingAfter?.contains("UserAgendaBridge", ignoreCase = true) == true

        val isPendingClarification =
            turn.pendingAfter?.contains("AskClarification", ignoreCase = true) == true

        val isConfirmFinalizeGate =
            (turn.phase == "CONFIRMING" || turn.phase == "SEALING") &&
                    (turn.pendingAfter?.contains("ConfirmStartSolving", ignoreCase = true) == true)

        val isConfirmRetakeGate =
            (turn.phase == "CONFIRMING" || turn.phase == "SEALING") &&
                    (turn.pendingAfter?.contains("ConfirmRetake", ignoreCase = true) == true)

        val isConfirmMismatchGate =
            (turn.phase == "CONFIRMING" || turn.phase == "SEALING") &&
                    (
                            turn.pendingAfter?.contains("ConfirmCell", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("ConfirmRegion", ignoreCase = true) == true
                            )

        val isConfirmNotUniqueGate =
            (turn.phase == "CONFIRMING" || turn.phase == "SEALING") &&
                    (
                            turn.pendingAfter?.contains("NotUnique", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("Unsolvable", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("Invalid", ignoreCase = true) == true
                            )

        val isConfirmExactMatchGate =
            (turn.phase == "CONFIRMING" || turn.phase == "SEALING") &&
                    !isConfirmRetakeGate &&
                    !isConfirmMismatchGate &&
                    !isConfirmNotUniqueGate &&
                    (
                            turn.pendingAfter?.contains("ConfirmValidate", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("VisualVerifyMatch", ignoreCase = true) == true
                            )

        val hasGridCandidateFacts =
            replyRequest.facts.any {
                when (it.type) {
                    FactBundleV1.Type.CANDIDATE_STATE_CELL,
                    FactBundleV1.Type.CELLS_WITH_N_CANDS_SET,
                    FactBundleV1.Type.BIVALUE_CELLS_SET,
                    FactBundleV1.Type.HOUSE_CANDIDATE_MAP,
                    FactBundleV1.Type.DIGIT_CANDIDATE_FREQUENCY,
                    FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                    FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1,
                    FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
                    FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
                    FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1 -> true
                    else -> false
                }
            }

        val hasGridValidationFacts =
            replyRequest.facts.any {
                when (it.type) {
                    FactBundleV1.Type.STRUCTURAL_VALIDITY,
                    FactBundleV1.Type.CONFLICT_SET,
                    FactBundleV1.Type.DUPLICATES_BY_HOUSE,
                    FactBundleV1.Type.UNRESOLVED_SET,
                    FactBundleV1.Type.MISMATCH_SET,
                    FactBundleV1.Type.OCR_CONFIDENCE_CELL,
                    FactBundleV1.Type.OCR_CONFIDENCE_SUMMARY,
                    FactBundleV1.Type.SEAL_STATUS,
                    FactBundleV1.Type.RECENT_MUTATION_RESULT -> true
                    else -> false
                }
            }

        val hasGridOcrTrustFacts =
            replyRequest.facts.any {
                when (it.type) {
                    FactBundleV1.Type.OCR_CONFIDENCE_CELL,
                    FactBundleV1.Type.OCR_CONFIDENCE_SUMMARY -> true
                    else -> false
                }
            }

        val hasGridContentsFacts =
            replyRequest.facts.any {
                when (it.type) {
                    FactBundleV1.Type.GRID_SNAPSHOT,
                    FactBundleV1.Type.CELL_STATUS_BUNDLE,
                    FactBundleV1.Type.HOUSE_STATUS_BUNDLE,
                    FactBundleV1.Type.HOUSES_COMPLETION_RANKING,
                    FactBundleV1.Type.DIGIT_LOCATIONS_BUNDLE -> true
                    else -> false
                }
            }

        val hasGridChangelogFacts =
            replyRequest.facts.any {
                when (it.type) {
                    FactBundleV1.Type.RECENT_MUTATION_RESULT -> true
                    else -> false
                }
            }

        val hasSolvingSupportFacts =
            replyRequest.facts.any {
                when (it.type) {
                    FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1,
                    FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1,
                    FactBundleV1.Type.HANDOVER_NOTE_V1,
                    FactBundleV1.Type.STORY_CONTEXT_V1,
                    FactBundleV1.Type.CTA_PACKET_V1,
                    FactBundleV1.Type.SOLVING_STEP_PACKET_V1,
                    FactBundleV1.Type.SETUP_REPLY_PACKET_V1,
                    FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1,
                    FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1 -> true
                    else -> false
                }
            }

        val detourDemandCategory =
            resolveDetourDemandCategoryFromFactsV1(replyRequest)

        return when {
            replyRequest.openingTurn ->
                PromptModuleDemandCategoryV1.ONBOARDING_OPENING

            isUserAgendaBridge ->
                when {
                    detourDemandCategory != null ->
                        detourDemandCategory

                    hasSolvingSupportFacts ->
                        PromptModuleDemandCategoryV1.SOLVING_ROUTE_CONTROL

                    hasGridCandidateFacts ->
                        PromptModuleDemandCategoryV1.GRID_CANDIDATE_ANSWER

                    hasGridOcrTrustFacts ->
                        PromptModuleDemandCategoryV1.GRID_OCR_TRUST_ANSWER

                    hasGridContentsFacts ->
                        PromptModuleDemandCategoryV1.GRID_CONTENTS_ANSWER

                    hasGridChangelogFacts ->
                        PromptModuleDemandCategoryV1.GRID_CHANGELOG_ANSWER

                    hasGridValidationFacts ->
                        PromptModuleDemandCategoryV1.GRID_VALIDATION_ANSWER

                    turn.phase == "SOLVING" ->
                        PromptModuleDemandCategoryV1.SOLVING_ROUTE_CONTROL

                    else ->
                        PromptModuleDemandCategoryV1.META_STATE_ANSWER
                }

            isPendingClarification ->
                PromptModuleDemandCategoryV1.PENDING_CLARIFICATION

            isConfirmFinalizeGate ->
                PromptModuleDemandCategoryV1.CONFIRM_FINALIZE_GATE

            isConfirmRetakeGate ->
                PromptModuleDemandCategoryV1.CONFIRM_RETAKE_GATE

            isConfirmMismatchGate ->
                PromptModuleDemandCategoryV1.CONFIRM_MISMATCH_GATE

            isConfirmNotUniqueGate ->
                PromptModuleDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE

            isConfirmExactMatchGate ->
                PromptModuleDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE

            detourDemandCategory != null ->
                detourDemandCategory

            turn.phase == "SOLVING" &&
                    turn.story?.canonicalPositionKind == "SETUP" ->
                PromptModuleDemandCategoryV1.SOLVING_SETUP

            turn.phase == "SOLVING" &&
                    turn.story?.canonicalPositionKind == "CONFRONTATION" ->
                PromptModuleDemandCategoryV1.SOLVING_CONFRONTATION

            turn.phase == "SOLVING" &&
                    (
                            turn.story?.canonicalPositionKind == "RESOLUTION_COMMIT" ||
                                    turn.story?.canonicalPositionKind == "RESOLUTION_POST_COMMIT"
                            ) ->
                PromptModuleDemandCategoryV1.SOLVING_RESOLUTION

            turn.phase == "SOLVING" &&
                    (turn.pendingAfter?.contains("ReturnToRoute", ignoreCase = true) == true) ->
                PromptModuleDemandCategoryV1.REPAIR_CONTRADICTION

            hasGridCandidateFacts ->
                PromptModuleDemandCategoryV1.GRID_CANDIDATE_ANSWER

            hasGridOcrTrustFacts ->
                PromptModuleDemandCategoryV1.GRID_OCR_TRUST_ANSWER

            hasGridContentsFacts ->
                PromptModuleDemandCategoryV1.GRID_CONTENTS_ANSWER

            hasGridChangelogFacts ->
                PromptModuleDemandCategoryV1.GRID_CHANGELOG_ANSWER

            hasGridValidationFacts ->
                PromptModuleDemandCategoryV1.GRID_VALIDATION_ANSWER

            turn.phase == "SOLVING" ->
                PromptModuleDemandCategoryV1.SOLVING_STAGE_ELABORATION

            turn.phase == "CONFIRMING" || turn.phase == "SEALING" ->
                PromptModuleDemandCategoryV1.GRID_VALIDATION_ANSWER

            else ->
                PromptModuleDemandCategoryV1.META_STATE_ANSWER
        }
    }

    private fun selectedPromptModulesForDemandCategoryV1(
        replyRequest: ReplyRequestV1,
        demandCategory: PromptModuleDemandCategoryV1
    ): List<PromptModuleV1> {
        return when (demandCategory) {
            PromptModuleDemandCategoryV1.SOLVING_SETUP -> {
                val setupPacket =
                    replyRequest.facts
                        .firstOrNull { it.type == FactBundleV1.Type.SETUP_REPLY_PACKET_V1 }
                        ?.payload

                when (setupPacket?.optString("setup_doctrine")) {
                    "LENS_FIRST" ->
                        listOf(PromptModuleV1.SETUP_LENS_FIRST_RULES)

                    "PATTERN_FIRST" ->
                        listOf(PromptModuleV1.SETUP_PATTERN_FIRST_RULES)

                    else -> emptyList()
                }
            }

            PromptModuleDemandCategoryV1.DETOUR_PROOF_CHALLENGE -> {
                listOf(PromptModuleV1.DETOUR_MOVE_PROOF_RULES) +
                        doctrineModulesForProofChallengeFactsV1(replyRequest)
            }

            PromptModuleDemandCategoryV1.DETOUR_TARGET_CELL_QUERY -> {
                listOf(PromptModuleV1.DETOUR_MOVE_PROOF_RULES)
            }

            PromptModuleDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY,
            PromptModuleDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH -> {
                listOf(PromptModuleV1.DETOUR_LOCAL_GRID_INSPECTION_RULES)
            }

            PromptModuleDemandCategoryV1.DETOUR_REASONING_CHECK -> {
                listOf(PromptModuleV1.DETOUR_USER_PROPOSAL_VERDICT_RULES)
            }

            PromptModuleDemandCategoryV1.SOLVING_RESOLUTION -> {
                val resolutionPacket =
                    replyRequest.facts
                        .firstOrNull { it.type == FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1 }
                        ?.payload

                when (resolutionPacket?.optString("resolution_profile")) {
                    "BASE_SINGLES_RESOLUTION" ->
                        listOf(PromptModuleV1.RESOLUTION_BASIC_RULES)

                    "SUBSETS_RESOLUTION",
                    "INTERSECTIONS_RESOLUTION",
                    "ADVANCED_PATTERN_RESOLUTION" ->
                        listOf(PromptModuleV1.RESOLUTION_ADVANCED_RULES)

                    else -> emptyList()
                }
            }

            else -> emptyList()
        }
    }

    private fun composeDeveloperPromptForDemandCategoryV1(
        replyRequest: ReplyRequestV1,
        demandCategory: PromptModuleDemandCategoryV1
    ): String {
        return when (demandCategory) {
            PromptModuleDemandCategoryV1.ONBOARDING_OPENING ->
                DeveloperPromptComposer.composeTick2OnboardingDeveloperPrompt()

            PromptModuleDemandCategoryV1.PENDING_CLARIFICATION ->
                DeveloperPromptComposer.composeTick2PendingClarificationDeveloperPrompt()

            PromptModuleDemandCategoryV1.CONFIRM_FINALIZE_GATE ->
                DeveloperPromptComposer.composeTick2ConfirmFinalizeDeveloperPrompt()

            PromptModuleDemandCategoryV1.CONFIRM_RETAKE_GATE ->
                DeveloperPromptComposer.composeTick2ConfirmRetakeDeveloperPrompt()

            PromptModuleDemandCategoryV1.CONFIRM_MISMATCH_GATE ->
                DeveloperPromptComposer.composeTick2ConfirmMismatchDeveloperPrompt()

            PromptModuleDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE ->
                DeveloperPromptComposer.composeTick2ConfirmNotUniqueDeveloperPrompt()

            PromptModuleDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE ->
                DeveloperPromptComposer.composeTick2ConfirmExactMatchDeveloperPrompt()

            PromptModuleDemandCategoryV1.CONFIRM_STATUS_SUMMARY ->
                DeveloperPromptComposer.composeTick2ConfirmStatusDeveloperPrompt()

            PromptModuleDemandCategoryV1.GRID_CANDIDATE_ANSWER ->
                DeveloperPromptComposer.composeTick2GridCandidateAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.GRID_OCR_TRUST_ANSWER ->
                DeveloperPromptComposer.composeTick2GridOcrTrustAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.GRID_CONTENTS_ANSWER ->
                DeveloperPromptComposer.composeTick2GridContentsAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.GRID_CHANGELOG_ANSWER ->
                DeveloperPromptComposer.composeTick2GridChangelogAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.GRID_VALIDATION_ANSWER ->
                DeveloperPromptComposer.composeTick2GridValidationAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.ASSISTANT_PAUSE_RESUME ->
                DeveloperPromptComposer.composeTick2AssistantPauseResumeDeveloperPrompt()

            PromptModuleDemandCategoryV1.VALIDATE_ONLY_OR_SOLVE_ONLY ->
                DeveloperPromptComposer.composeTick2ValidateOnlyOrSolveOnlyDeveloperPrompt()

            PromptModuleDemandCategoryV1.FOCUS_REDIRECT ->
                DeveloperPromptComposer.composeTick2FocusRedirectDeveloperPrompt()

            PromptModuleDemandCategoryV1.PREFERENCE_CHANGE ->
                DeveloperPromptComposer.composeTick2PreferenceChangeDeveloperPrompt()

            PromptModuleDemandCategoryV1.META_STATE_ANSWER ->
                DeveloperPromptComposer.composeTick2MetaStateAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.CAPABILITY_ANSWER ->
                DeveloperPromptComposer.composeTick2CapabilityAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.GLOSSARY_ANSWER ->
                DeveloperPromptComposer.composeTick2GlossaryAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.UI_HELP_ANSWER ->
                DeveloperPromptComposer.composeTick2UiHelpAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.COORDINATE_HELP_ANSWER ->
                DeveloperPromptComposer.composeTick2CoordinateHelpAnswerDeveloperPrompt()

            PromptModuleDemandCategoryV1.SMALL_TALK_BRIDGE ->
                DeveloperPromptComposer.composeTick2SmallTalkBridgeDeveloperPrompt()

            PromptModuleDemandCategoryV1.FREE_TALK_NON_GRID ->
                DeveloperPromptComposer.composeTick2FreeTalkNonGridDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_TARGET_CELL_QUERY ->
                DeveloperPromptComposer.composeTick2DetourTargetCellQueryDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY ->
                DeveloperPromptComposer.composeTick2DetourNeighborCellQueryDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_REASONING_CHECK ->
                DeveloperPromptComposer.composeTick2DetourReasoningCheckDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE ->
                DeveloperPromptComposer.composeTick2DetourAlternativeTechniqueDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH ->
                DeveloperPromptComposer.composeTick2DetourLocalMoveSearchDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_ROUTE_COMPARISON ->
                DeveloperPromptComposer.composeTick2DetourRouteComparisonDeveloperPrompt()

            PromptModuleDemandCategoryV1.DETOUR_PROOF_CHALLENGE ->
                DeveloperPromptComposer.composeTick2DetourProofChallengeDeveloperPrompt()

            PromptModuleDemandCategoryV1.SOLVING_SETUP ->
                DeveloperPromptComposer.composeTick2SolvingSetupDeveloperPrompt()

            PromptModuleDemandCategoryV1.SOLVING_CONFRONTATION ->
                DeveloperPromptComposer.composeTick2SolvingConfrontationDeveloperPrompt()

            PromptModuleDemandCategoryV1.SOLVING_RESOLUTION ->
                DeveloperPromptComposer.composeTick2SolvingResolutionDeveloperPrompt()

            PromptModuleDemandCategoryV1.REPAIR_CONTRADICTION ->
                DeveloperPromptComposer.composeTick2RepairDeveloperPrompt()

            PromptModuleDemandCategoryV1.SOLVING_STAGE_ELABORATION ->
                DeveloperPromptComposer.composeTick2SolvingStageElaborationDeveloperPrompt()

            PromptModuleDemandCategoryV1.SOLVING_ROUTE_CONTROL -> {
                DeveloperPromptComposer.composeTick2DeveloperPrompt(
                    developerPreamble = TICK2_DEVELOPER_PREAMBLE_V1,
                    stageBlock = ""
                )
            }

            else -> {
                DeveloperPromptComposer.composeTick2DeveloperPrompt(
                    developerPreamble = TICK2_DEVELOPER_PREAMBLE_V1,
                    stageBlock = ""
                )
            }
        }
    }

    private fun projectedFactsArrayFromPlanV1(
        planResult: ReplyAssemblyPlannerV1.PlanResultV1
    ): org.json.JSONArray {
        val out = org.json.JSONArray()
        for (entry in planResult.projectedChannels) {
            out.put(
                JSONObject().apply {
                    put("type", entry.channel.name)
                    put("payload", entry.payload)
                }
            )
        }
        return out
    }

    private fun buildPlanLockedPacketCenteredDetourUserMessageV1(
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1
    ): String {
        val projectedFacts = projectedFactsArrayFromPlanV1(planResult)

        runCatching {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TICK2_PACKET_CENTERED_DETOUR_PLAN_LOCK_V1",
                    "turn_id" to replyRequest.turn.turnId,
                    "plan_demand_category" to planResult.plan.demand.category.name,
                    "selected_prompt_modules" to planResult.plan.selectedPromptModules.map { it.name },
                    "selected_channels" to planResult.plan.selectedChannels.map { it.name },
                    "projected_fact_count" to projectedFacts.length()
                )
            )
        }

        val shaped = pruneLegacyFatFieldsForProjectedBody(
            JSONObject(replyRequest.toJsonString()).apply {
                put("facts", projectedFacts)
            },
            preserveTalliesAndRecentTurns = true
        )

        return shaped.toString()
    }



    /**
     * Constitutional owner-first body shaping path.
     *
     * Order of truth:
     *  1) sovereign turn owner
     *  2) boundary state
     *  3) owner-compatible demand family
     *  4) owner-compatible projection lane
     *
     * Route-return is an ornament only.
     * Legacy full-body fallback is compatibility only.
     *
     * Must-pass regression hooks:
     *  - DETOUR_OWNER_PRESERVES_MEMORY_TALLIES
     *  - ROUTE_JUMP_OWNER_USES_ROUTE_JUMP_PROJECTION
     *  - REPAIR_OWNER_USES_REPAIR_PROJECTION
     *  - BOUNDARY_RELEASE_CAN_FEED_NEXT_SETUP_PROJECTION
     */
    private fun buildTick2UserMessageV2(
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null
    ): String {
        val turn = replyRequest.turn


        if (shouldUsePlanLockedPacketCenteredDetourPathV1(planResult)) {
            return buildPlanLockedPacketCenteredDetourUserMessageV1(
                replyRequest = replyRequest,
                planResult = requireNotNull(planResult)
            )
        }

        val isOnboarding =
            replyRequest.openingTurn

        val authorityOwner = turn.turnAuthorityOwner.orEmpty()
        val boundaryStatus = turn.turnBoundaryStatus.orEmpty()

        val isAuthorityAppRouteOwner =
            authorityOwner == "APP_ROUTE_OWNER"

        val isAuthorityUserDetourOwner =
            authorityOwner == "USER_DETOUR_OWNER"

        val isAuthorityUserRouteJumpOwner =
            authorityOwner == "USER_ROUTE_JUMP_OWNER"

        val isAuthorityRepairOwner =
            authorityOwner == "REPAIR_OWNER"

        val isBoundaryReleasedToNextStep =
            boundaryStatus == "RELEASED_TO_NEXT_STEP"

        val isConfirmingValidationSummary =
            turn.phase == "CONFIRMING" &&
                    (
                            turn.pendingAfter?.contains("ConfirmValidate", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("ConfirmStartSolving", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("VisualVerifyMatch", ignoreCase = true) == true ||
                                    turn.pendingAfter?.contains("ConfirmRetake", ignoreCase = true) == true
                            )

        val isSolvingSetup =
            ENABLE_SOLVING_SETUP_PACKET_V1 &&
                    (isAuthorityAppRouteOwner || isBoundaryReleasedToNextStep) &&
                    turn.phase == "SOLVING" &&
                    turn.story?.stage == "SETUP"

        val isSolvingConfrontation =
            isAuthorityAppRouteOwner &&
                    turn.phase == "SOLVING" &&
                    turn.story?.stage == "CONFRONTATION"

        val isSolvingResolution =
            isAuthorityAppRouteOwner &&
                    turn.phase == "SOLVING" &&
                    turn.story?.stage == "RESOLUTION"

        val shouldUseDetourProjection =
            isAuthorityUserDetourOwner

        val shouldUseRouteJumpProjection =
            isAuthorityUserRouteJumpOwner

        val shouldUseRepairProjection =
            isAuthorityRepairOwner

        if (!isOnboarding &&
            !shouldUseDetourProjection &&
            !shouldUseRouteJumpProjection &&
            !isConfirmingValidationSummary &&
            !isSolvingSetup &&
            !isSolvingConfrontation &&
            !isSolvingResolution &&
            !shouldUseRepairProjection
        ) {
            return replyRequest.toJsonString()
        }

        val projectionLane =
            when {
                isOnboarding -> "onboarding"
                shouldUseDetourProjection -> "user_detour"
                shouldUseRouteJumpProjection -> "route_jump"
                isConfirmingValidationSummary -> "confirming"
                isSolvingSetup -> "solving_setup"
                isSolvingConfrontation -> "solving_confrontation"
                isSolvingResolution -> "solving_resolution"
                shouldUseRepairProjection -> "repair"
                else -> "raw_passthrough"
            }

        if (isAuthorityUserDetourOwner && (isSolvingSetup || isSolvingConfrontation || isSolvingResolution)) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "PROJECTION_AUTHORITY_CONFLICT_V1",
                    "turn_id" to turn.turnId,
                    "authority_owner" to authorityOwner,
                    "boundary_status" to boundaryStatus,
                    "winner" to projectionLane,
                    "suppressed" to "solving_projection",
                    "reason" to "detour_owner_prevents_solving_projection_lane"
                )
            )
        }

        if (isAuthorityUserRouteJumpOwner && turn.turnRouteReturnAllowed) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "PROJECTION_AUTHORITY_CONFLICT_V1",
                    "turn_id" to turn.turnId,
                    "authority_owner" to authorityOwner,
                    "boundary_status" to boundaryStatus,
                    "winner" to projectionLane,
                    "suppressed" to "detour_route_return_bias",
                    "reason" to "route_jump_projection_does_not_use_detour_return_bridge"
                )
            )
        }

        val projectedFacts =
            when {
                isOnboarding ->
                    ReplySupplyProjectorsV1.projectFactsForOnboardingDemand(replyRequest)

                shouldUseDetourProjection ->
                    ReplySupplyProjectorsV1.projectFactsForUserDetourDemand(replyRequest)

                shouldUseRouteJumpProjection ->
                    ReplySupplyProjectorsV1.projectFactsForRouteJumpDemand(replyRequest)

                isConfirmingValidationSummary ->
                    ReplySupplyProjectorsV1.projectFactsForConfirmingDemand(replyRequest)

                isSolvingSetup ->
                    ReplySupplyProjectorsV1.projectFactsForSetupDemand(replyRequest)

                isSolvingConfrontation ->
                    ReplySupplyProjectorsV1.projectFactsForConfrontationDemand(replyRequest)

                isSolvingResolution ->
                    ReplySupplyProjectorsV1.projectFactsForResolutionDemand(replyRequest)

                shouldUseRepairProjection ->
                    ReplySupplyProjectorsV1.projectFactsForRepairDemand(replyRequest)

                else ->
                    FactBundleV1.jsonArray(replyRequest.facts)
            }

        val preserveTalliesAndRecentTurns =
            shouldUseDetourProjection || shouldUseRepairProjection

        ConversationTelemetry.emit(
            mapOf(
                "type" to "PROJECTION_LANE_TRACE_V1",
                "turn_id" to turn.turnId,
                "authority_owner" to authorityOwner,
                "boundary_status" to boundaryStatus,
                "projection_lane" to projectionLane,
                "preserve_tallies_and_recent_turns" to preserveTalliesAndRecentTurns,
                "facts_count" to projectedFacts.length()
            )
        )

        ConversationTelemetry.emit(
            mapOf(
                "type" to "PROJECTION_INVARIANT_V1",
                "turn_id" to turn.turnId,
                "authority_owner" to authorityOwner,
                "boundary_status" to boundaryStatus,
                "invariant" to "DETOUR_OR_REPAIR_MUST_PRESERVE_MEMORY_WHEN_PROJECTION_IS_SHAPED",
                "ok" to !((shouldUseDetourProjection || shouldUseRepairProjection) && !preserveTalliesAndRecentTurns),
                "detail" to "detour=$shouldUseDetourProjection repair=$shouldUseRepairProjection preserveTallies=$preserveTalliesAndRecentTurns"
            )
        )

        val shaped = pruneLegacyFatFieldsForProjectedBody(
            JSONObject(replyRequest.toJsonString()).apply {
                put("facts", projectedFacts)
            },
            preserveTalliesAndRecentTurns = preserveTalliesAndRecentTurns
        )

        return shaped.toString()
    }



    /**
     * Demand-specific Tick2 system prompt selection.
     *
     * Runtime selection now prefers explicit modern demand families.
     * Transitional legacy full selection is no longer used here.
     */
    private fun buildTick2SystemPromptV2(
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null
    ): String {
        val turn = replyRequest.turn


        if (shouldUsePlanLockedPacketCenteredDetourPathV1(planResult)) {
            val lockedPlan = requireNotNull(planResult).plan
            val lockedDemandCategory =
                promptModuleDemandCategoryFromReplyDemandV1(lockedPlan.demand.category)


            runCatching {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TICK2_DETOUR_SYSTEM_MODULE_LOCK_V1",
                        "turn_id" to turn.turnId,
                        "plan_demand_category" to lockedPlan.demand.category.name,
                        "selected_prompt_modules" to lockedPlan.selectedPromptModules.map { it.name }
                    )
                )
            }

            val basePrompt = PromptModulesV1.composeSystemPromptForDemand(
                demandCategory = lockedDemandCategory,
                modules = TICK2_SYSTEM_PROMPT_MODULES_V1,
                selectedPromptModules = lockedPlan.selectedPromptModules
            )

            return if (hasAnyDetourPacketV1(replyRequest)) {
                listOf(
                    basePrompt.trim(),
                    detourPromptAppendixV1()
                ).joinToString("\n\n").trim()
            } else {
                basePrompt
            }
        }

        val demandCategory =
            authoritativeReplyDemandCategoryFromRequestV1(
                replyRequest = replyRequest,
                planResult = planResult
            )?.let(::promptModuleDemandCategoryFromReplyDemandV1)
                ?: fallbackPromptModuleDemandCategoryFromStateV1(replyRequest)

        val selectedPromptModules =
            selectedPromptModulesForDemandCategoryV1(
                replyRequest = replyRequest,
                demandCategory = demandCategory
            )

        val basePrompt = PromptModulesV1.composeSystemPromptForDemand(
            demandCategory = demandCategory,
            modules = TICK2_SYSTEM_PROMPT_MODULES_V1,
            selectedPromptModules = selectedPromptModules
        )

        return if (hasAnyDetourPacketV1(replyRequest)) {
            listOf(
                basePrompt.trim(),
                detourPromptAppendixV1()
            ).joinToString("\n\n").trim()
        } else {
            basePrompt
        }
    }

// -------------------------------------------------------------------------
// Tick 1 (Intent Envelope) — coordinator entrypoint
// -------------------------------------------------------------------------

    suspend fun sendIntentEnvelopeV1(
        mode: String,
        phase: String,
        pendingBefore: String?,
        pendingExpectedAnswerKind: String?,
        pendingTargetCell: String?,
        focusCell: String?,
        lastAssistantQuestionKey: String?,
        userText: String,
        userTallyJson: String,
        assistantTallyJson: String,
        recentTurnsJson: String,
        discourseStateJson: String? = null,
        telemetryCtx: ModelCallTelemetryCtx? = null
    ): com.contextionary.sudoku.conductor.policy.IntentEnvelopeV1 {

        val developerPrompt = buildString {
            appendLine("Tick-1 instructions:")
            appendLine("- The user message is TurnContextV1 JSON (schema='TurnContextV1').")
            appendLine("- Read user_text as the utterance.")

            appendLine("- TurnContextV1.canonical_solving_position_kind is a TOP-LEVEL field. Do not look for story.canonical_position_kind.")
            appendLine("- awaited_assistant_answer.owner and awaited_assistant_answer.followup_disposition are authoritative.")
            appendLine("- TurnContextV1.solving_handoff, when present with authority='STRUCTURED_APP_STATE', is authoritative structured app truth and outranks ambiguous wording in recent_turns.")
            appendLine("- If solving_handoff.commit_already_applied == true, do NOT interpret generic assent as a request to apply or re-apply the committed move.")
            appendLine("- If solving_handoff.assistant_cta_kind == 'CONTINUE_ROUTE', generic assent like 'yes', 'yeah', 'ok', 'okay', 'go ahead', 'please go ahead', 'sure', 'continue', 'next', 'keep going', 'move on' should default to solving_handoff.generic_assent_default_intent unless the user clearly opens a detour.")
            appendLine("- Under solving_handoff.detour_override_rule == 'ONLY_IF_EXPLICIT', classify a detour only when the user clearly asks a question, reports a board-sync problem, gives a concrete edit instruction, or otherwise explicitly departs from the main route.")
            appendLine("- If awaited_assistant_answer.owner == 'USER_AGENDA_OWNER' or pending.pending_before contains 'UserAgendaBridge', treat the turn as a USER-OWNED DETOUR, not an app-road continuation.")
            appendLine("- In user-owned detours, explicit user scope wins over focus, pending, recent_turns, and discourse_state_json.")
            appendLine("- In user-owned detours, use focus_cell only when the utterance is TRULY deictic ('it', 'that cell', 'there', 'this one') and no stronger explicit scope is present.")
            appendLine("- Respect TurnContextV1.focus_coreference_policy and TurnContextV1.recent_turns_policy.")
            appendLine("- If TurnContextV1.focus_coreference_policy == 'STRICT_DEICTIC_ONLY', only explicit demonstratives like 'it', 'that cell', 'this one', or 'there' may inherit focus_cell. Generic noun phrases like 'the cell', 'the spot', or 'the square' are NOT enough.")
            appendLine("- If TurnContextV1.recent_turns_policy == 'DEICTIC_ONLY_NO_CELL_COMPLETION', recent_turns may help maintain topic family or digit continuity, but must NOT supply a concrete cell unless the current utterance is truly deictic and contains no conflicting scope.")
            appendLine("- If the utterance contains both a row reference and a column reference, prefer a CELL target over a HOUSE target, even if ASR is noisy.")
            appendLine("- If the utterance partially specifies a cell and ASR is noisy, preserve a cell hypothesis only when at least one coordinate anchor is recoverable in the CURRENT utterance (explicit row, explicit column, or explicit rXcY). Lower confidence if needed.")
            appendLine("- Do NOT complete a fully missing cell from recent_turns alone in a user-owned detour.")
            appendLine("- If there is no recoverable coordinate anchor and no true deictic reference, leave cell in missing rather than silently importing a prior cell from focus or recent_turns.")
            appendLine("- Use pending/focus/last_assistant_question_key/recent_turns/tally for coreference only under those rules.")
            appendLine("- Also use discourse_state_json (App-owned) to resolve 'it/that/there/continue' only when allowed by the detour rules above.")

            appendLine("- discourse_state_json: ${discourseStateJson ?: "null"}")
            appendLine("- Output ONLY IntentEnvelopeV1 JSON. No prose.")
            appendLine()
            appendLine(TICK1_FEWSHOTS_INTENT_ENVELOPE_V1)
        }.trim()

        // --- Preferred: your llmClient has a dedicated IntentEnvelope method ---
        // If your method name differs, rename ONLY this call.
        return llmClient.sendIntentEnvelope(
            systemPrompt = TICK1_SYSTEM_INTENT_ENVELOPE_V1,
            developerPrompt = developerPrompt,
            userMessage = userText,
            telemetryCtx = telemetryCtx
        )


    }

// -------------------------------------------------------------------------
// Tick 2 (Reply Generate) — coordinator entrypoint
// -------------------------------------------------------------------------

    suspend fun sendReplyGenerateV1(
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null,
        telemetryCtx: ModelCallTelemetryCtx? = null
    ): String {

        // Phase 5 — inject TEACHING_CARD when we have a technique id in SOLVING facts.
        val req2 = injectTeachingCardIfNeeded(replyRequest)

        val developerPrompt = buildTick2DeveloperPromptV2(
            replyRequest = req2,
            planResult = planResult
        )

        return llmClient.sendReplyGenerate(
            systemPrompt = buildTick2SystemPromptV2(
                replyRequest = req2,
                planResult = planResult
            ),
            developerPrompt = developerPrompt,
            userMessage = buildTick2UserMessageV2(
                replyRequest = req2,
                planResult = planResult
            ),
            telemetryCtx = telemetryCtx
        )
    }




    private fun injectTeachingCardIfNeeded(req: ReplyRequestV1): ReplyRequestV1 {
        // Only meaningful for SOLVING (your Phase-5 rule in the Tick2 system prompt).
        val phase = req.turn.phase.trim().uppercase()
        if (phase != "SOLVING") return req

        // If already present, do nothing.
        if (req.facts.any { it.type == FactBundleV1.Type.TEACHING_CARD }) return req

        val techniqueId = extractTechniqueIdFromFacts(req.facts) ?: return req

        // Prefer language from PREFERENCES_SNAPSHOT when present.
        val preferredLang = extractPreferredLanguageFromFacts(req.facts)

        // Try KB; if unknown id, inject a safe fallback so Tick2 can still answer “what is this technique?”
        val card =
            com.contextionary.sudoku.conductor.solving.TechniqueCards.bundleFor(techniqueId, preferredLang)
                ?: com.contextionary.sudoku.conductor.solving.TechniqueCards.fallbackBundleFor(techniqueId, preferredLang)
                ?: return req

        val facts2 = req.facts.toMutableList()
        facts2.add(card)

        return req.copy(facts = facts2)
    }

    private fun extractPreferredLanguageFromFacts(facts: List<FactBundleV1>): String? {
        val p = facts.firstOrNull { it.type == FactBundleV1.Type.PREFERENCES_SNAPSHOT }?.payload ?: return null
        val lang = p.optString("language", "").trim()
        return lang.takeIf { it.isNotBlank() }
    }

    private fun extractTechniqueIdFromFacts(facts: List<FactBundleV1>): String? {
        // 1) Preferred: SOLVING_STEP_PACKET_V1
        facts.firstOrNull { it.type == FactBundleV1.Type.SOLVING_STEP_PACKET_V1 }?.payload?.let { j ->
            // Common shapes:
            // - technique_id: "singles-1"
            // - technique: { id: "..."} or { name:"..."} depending on adapter
            val direct = j.optString("technique_id", "").trim()
            if (direct.isNotBlank()) return direct

            val techniqueObj = j.optJSONObject("technique")
            if (techniqueObj != null) {
                val id = techniqueObj.optString("id", "").trim()
                if (id.isNotBlank()) return id
                val name = techniqueObj.optString("technique_id", "").trim()
                if (name.isNotBlank()) return name
            }

            // Some packets nest under engine/step:
            j.optJSONObject("step")?.optJSONObject("engine")?.let { eng ->
                val id = eng.optString("technique_id", "").trim()
                if (id.isNotBlank()) return id
            }
        }

        // 2) Fallback: TECHNIQUE_FINDINGS (if present)
        facts.firstOrNull { it.type == FactBundleV1.Type.TECHNIQUE_FINDINGS }?.payload?.let { j ->
            val id = j.optString("technique_id", "").trim()
            if (id.isNotBlank()) return id
            val name = j.optString("technique", "").trim()
            if (name.isNotBlank()) return name
        }

        return null
    }

    private data class ClassifiedInput(
        val source: String,          // "voice" | "text" | "ui_event" | "synthetic"
        val rawText: String,
        val normalizedText: String,
        val pendingType: String?,
        val pendingIdx: Int?
    )

    private fun classifyUserInput(userMessage: String): ClassifiedInput {
        val raw = userMessage
        val isEvent = raw.trimStart().startsWith("[EVENT]", ignoreCase = true)

        if (!isEvent) {
            return ClassifiedInput(
                source = "text",
                rawText = raw,
                normalizedText = raw.trim(),
                pendingType = null,
                pendingIdx = null
            )
        }

        val pendingType = Regex("""pending[_: ]([a-zA-Z0-9_]+)""")
            .find(raw)?.groupValues?.getOrNull(1)

        val pendingIdx = Regex("""idx=([0-9]+)""")
            .find(raw)?.groupValues?.getOrNull(1)?.toIntOrNull()

        val normalized = Regex("""raw='([^']*)'""")
            .find(raw)?.groupValues?.getOrNull(1)
            ?.trim()
            ?: raw

        return ClassifiedInput(
            source = "ui_event",
            rawText = raw,
            normalizedText = normalized,
            pendingType = pendingType,
            pendingIdx = pendingIdx
        )
    }

    private enum class ValidationMode { GLOBAL, CELL_CHECK }
    private enum class PendingPolicy { CANCEL_ON_GLOBAL_VALIDATION, NONE }

    private fun computeValidationMode(g: LLMGridState): ValidationMode {
        val solvable = (g.solvability == "unique" || g.solvability == "multiple")
        return if (solvable && g.conflictCells.isEmpty() && g.mismatchCells.isEmpty()) {
            ValidationMode.GLOBAL
        } else {
            ValidationMode.CELL_CHECK
        }
    }

    private fun computePendingPolicy(mode: ValidationMode): PendingPolicy =
        if (mode == ValidationMode.GLOBAL) PendingPolicy.CANCEL_ON_GLOBAL_VALIDATION else PendingPolicy.NONE



    private fun narrativeStageScopeV1(
        expectedAnswerKind: String?,
        atomIndex: Int?,
        atomsCount: Int?
    ): String {
        val n = atomsCount ?: 0
        val idx = atomIndex ?: 0
        val lastProof = (n - 2).coerceAtLeast(0)

        return when {
            expectedAnswerKind == "STORY_CTA_SETUP" -> "SETUP_ONLY"
            expectedAnswerKind == "STORY_CTA_REVEAL" -> "RESOLUTION_ONLY"
            idx <= 0 -> "SETUP_ONLY"
            idx in 1..lastProof -> "CONFRONTATION_ONLY"
            else -> "RESOLUTION_ONLY"
        }
    }




    private fun deriveStageScopeFromReplyRequestJsonV1(replyRequest: ReplyRequestV1): String {
        return runCatching {
            val root = org.json.JSONObject(replyRequest.toJsonString())

            val turn = root.optJSONObject("turn") ?: org.json.JSONObject()
            val story = turn.optJSONObject("story") ?: org.json.JSONObject()
            val pending = turn.optJSONObject("pending") ?: org.json.JSONObject()

            val canonicalPositionKind =
                story.optString("canonical_position_kind", "").trim().uppercase()
            val storyStage = story.optString("stage", "").trim().uppercase()

            if (canonicalPositionKind.isBlank()) {
                if (storyStage == "SETUP") return@runCatching "SETUP_ONLY"
                if (storyStage == "CONFRONTATION") return@runCatching "CONFRONTATION_ONLY"
                if (storyStage == "RESOLUTION") return@runCatching "RESOLUTION_ONLY"
            }

            val expectedAnswerKind = pending.optString("expected_answer_kind", "").trim()
                .ifBlank { null }

            val facts = root.optJSONArray("facts") ?: org.json.JSONArray()

            var atomIndex: Int? = null
            var atomsCount: Int? = null

            for (i in 0 until facts.length()) {
                val f = facts.optJSONObject(i) ?: continue
                val type = f.optString("type", "").trim()
                if (type != "STORY_CONTEXT_V1") continue

                val payload = f.optJSONObject("payload") ?: org.json.JSONObject()

                val atomsInScope = payload.optJSONArray("atoms_in_scope_this_turn")
                if (atomsInScope != null && atomsInScope.length() > 0) {
                    atomIndex = atomsInScope.optInt(0, atomIndex ?: 0)
                }

                atomsCount = payload.optInt("atoms_count", atomsCount ?: 0)
                break
            }

            narrativeStageScopeV1(
                expectedAnswerKind = expectedAnswerKind,
                atomIndex = atomIndex,
                atomsCount = atomsCount
            )
        }.getOrElse { "UNKNOWN_STAGE_SCOPE" }
    }

    private fun buildTick2DeveloperPromptV2(
        replyRequest: ReplyRequestV1,
        planResult: ReplyAssemblyPlannerV1.PlanResultV1? = null
    ): String {
        val turn = replyRequest.turn

        if (shouldUsePlanLockedPacketCenteredDetourPathV1(planResult)) {
            val lockedCategory = requireNotNull(planResult).plan.demand.category

            runCatching {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TICK2_DETOUR_DEVELOPER_BRANCH_LOCK_V1",
                        "turn_id" to turn.turnId,
                        "plan_demand_category" to lockedCategory.name
                    )
                )
            }

            return when (lockedCategory) {
                ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY ->
                    DeveloperPromptComposer.composeTick2DetourTargetCellQueryDeveloperPrompt()

                ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY ->
                    DeveloperPromptComposer.composeTick2DetourNeighborCellQueryDeveloperPrompt()

                ReplyDemandCategoryV1.DETOUR_REASONING_CHECK ->
                    DeveloperPromptComposer.composeTick2DetourReasoningCheckDeveloperPrompt()

                ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE ->
                    DeveloperPromptComposer.composeTick2DetourAlternativeTechniqueDeveloperPrompt()

                ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH ->
                    DeveloperPromptComposer.composeTick2DetourLocalMoveSearchDeveloperPrompt()

                ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON ->
                    DeveloperPromptComposer.composeTick2DetourRouteComparisonDeveloperPrompt()

                ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE ->
                    DeveloperPromptComposer.composeTick2DetourProofChallengeDeveloperPrompt()

                else ->
                    DeveloperPromptComposer.composeTick2DetourProofChallengeDeveloperPrompt()
            }
        }

        val demandCategory =
            authoritativeReplyDemandCategoryFromRequestV1(
                replyRequest = replyRequest,
                planResult = planResult
            )?.let(::promptModuleDemandCategoryFromReplyDemandV1)
                ?: fallbackPromptModuleDemandCategoryFromStateV1(replyRequest)

        return composeDeveloperPromptForDemandCategoryV1(
            replyRequest = replyRequest,
            demandCategory = demandCategory
        )
    }






    // -------------------------
    // FREE TALK (unchanged logic; telemetry gated)
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
        val histMsgs = builtMessages.drop(2)

        fun norm(s: String) = s.trim().replace(Regex("\\s+"), " ")
        val trimmed = histMsgs.toMutableList()
        if (trimmed.isNotEmpty()) {
            val last = trimmed.last()
            val isUser = last.role.name.equals("USER", ignoreCase = true)
            if (isUser && norm(last.content) == norm(currentUserMessage)) {
                trimmed.removeAt(trimmed.size - 1)
            }
        }

        val pairs = trimmed.mapNotNull { m ->
            val r = m.role.name.lowercase()
            if (r != "user" && r != "assistant") null
            else {
                val c = m.content.replace("\r\n", "\n").trimEnd()
                if (c.isBlank()) null else (r to c)
            }
        }

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

        t { ConversationTelemetry.emit(mapOf("type" to "DEV_PROMPT_FREE_TALK", "chars" to prompt.length)) }
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

        val lastAssistantIdx = histMsgs.indexOfLast { isAssistant(it) }
        val secondLastAssistantIdx =
            if (lastAssistantIdx > 0) {
                histMsgs.subList(0, lastAssistantIdx).indexOfLast { isAssistant(it) }
            } else {
                -1
            }

        val keepFrom = when {
            secondLastAssistantIdx >= 0 -> secondLastAssistantIdx
            lastAssistantIdx >= 0 -> lastAssistantIdx
            else -> histMsgs.size
        }

        val HEAD_USER_LIMIT = 8
        val headUsersAll = histMsgs.take(keepFrom).filter { isUser(it) }
        val head = if (headUsersAll.size > HEAD_USER_LIMIT) headUsersAll.takeLast(HEAD_USER_LIMIT) else headUsersAll
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

        t {
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
        }

        val resp = llmClient.chatFreeTalk(sys, dev, userMessage)

        lifecycle.commitAssistant(sessionId = sessionId, turnId = committedUser.turnId, text = resp.assistant_message)
        lifecycle.finalizeTurn(sessionId = sessionId, turnId = committedUser.turnId)

        t {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "PROMPT_HASH_USED",
                    "session_id" to sessionId,
                    "turn_id" to committedUser.turnId,
                    "persona_hash" to built.personaHash,
                    "prompt_hash" to built.promptHash
                )
            )
        }

        t {
            ConversationTelemetry.emitLlmResponseDigest(
                mode = "FREE_TALK",
                model = null,
                convoSessionId = sessionId,
                turnId = committedUser.turnId,
                assistantText = resp.assistant_message
            )
        }

        return resp.assistant_message
    }

    suspend fun freeTalk(
        @Suppress("UNUSED_PARAMETER") systemPrompt: String,
        profile: UserProfile,
        userMessage: String
    ): String {
        val snap = PlayerProfileSnapshot(
            name = null,
            locale = null,
            favoriteDifficulty = null,
            interests = emptyList()
        )
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
        t { ConversationTelemetry.emit(mapOf("type" to "CLUES_BEGIN", "transcript_chars" to transcript.length)) }
        val sys = buildClueExtractionSystemPrompt()
        val dev = buildClueExtractionDeveloperPrompt()
        val resp = llmClient.extractClues(sys, dev, transcript)
        t { ConversationTelemetry.emit(mapOf("type" to "CLUES_OK", "count" to resp.clues.size)) }
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

        t {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "RECOVERY_RESULT",
                    "session_id" to sessionId,
                    "decision" to r.decision.name,
                    "affected_turn_id" to (r.affectedTurnId ?: -1L),
                    "note" to (r.note ?: "")
                )
            )
        }
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
    // buildGridContextV1(...) (your v4.2 block) — cleaned for sanity:
    // - remove duplicate validationMode/pendingPolicy shadowing
    // - keep pure string building (no coordinator telemetry)
    // ----------------------------------------------------------------
    private fun buildGridContextV1(
        gridState: LLMGridState,
        phase: GridPhase,
        engineStepSummary: String?
    ): String {
        val digits = gridState.correctedGrid
        val rows = formatGridRows81(digits)

        val validationMode = computeValidationMode(gridState)
        val pendingPolicy = computePendingPolicy(validationMode)

        fun maskToList(mask: Int): List<Int> {
            if (mask == 0) return emptyList()
            val out = mutableListOf<Int>()
            for (d in 1..9) if ((mask and (1 shl (d - 1))) != 0) out += d
            return out
        }

        fun countBits(mask: Int): Int = Integer.bitCount(mask and 0x1FF)
        fun rc(idx: Int): String = "r${idx / 9 + 1}c${idx % 9 + 1}"

        val confirmed = gridState.confirmedCells.distinct().sorted()
        fun isConfirmed(idx: Int): Boolean = confirmed.contains(idx)

        val givenCount = (0 until 81).count { gridState.truthIsGiven[it] }
        val solCount = (0 until 81).count { gridState.truthIsSolution[it] }

        val totalCandidateMarks: Int =
            if (phase == GridPhase.SOLVING) 0
            else (0 until 81).sumOf { countBits(gridState.candidateMask81[it]) }

        val candByRow: IntArray =
            if (phase == GridPhase.SOLVING) IntArray(9)
            else IntArray(9).also { arr ->
                for (idx in 0 until 81) {
                    val r = idx / 9
                    val n = countBits(gridState.candidateMask81[idx])
                    arr[r] += n
                }
            }

        val candByCol: IntArray =
            if (phase == GridPhase.SOLVING) IntArray(9)
            else IntArray(9).also { arr ->
                for (idx in 0 until 81) {
                    val c = idx % 9
                    val n = countBits(gridState.candidateMask81[idx])
                    arr[c] += n
                }
            }

        val givensGrid = IntArray(81) { idx -> if (gridState.truthIsGiven[idx]) digits[idx] else 0 }
        val userSolGrid = IntArray(81) { idx -> if (gridState.truthIsSolution[idx]) digits[idx] else 0 }
        val givensRows = formatGridRows81(givensGrid)
        val userSolRows = formatGridRows81(userSolGrid)
        val deducedRows = gridState.deducedSolutionGrid?.let { formatGridRows81(it) }

        val candLines: String? =
            if (phase == GridPhase.SOLVING) null
            else buildString {
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
        // CONFLICTS_DETAILS (authoritative; computed from CURRENT_DISPLAY)
        // ------------------------------------------------------------
        data class HouseConflict(val houseType: String, val houseId: Int, val digit: Int, val indices: List<Int>)

        fun computeConflictsFromDisplay(): List<HouseConflict> {
            val out = mutableListOf<HouseConflict>()

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
        // RECOMMENDED_NEXT_CHECK — confirmation-aware policy
        // ------------------------------------------------------------
        data class NextCheck(val idx: Int, val priority: String, val reason: String)

        fun pickRecommendedNextCheck(): NextCheck? {
            fun pickBest(cands: List<Int>, label: String): NextCheck? {
                val filtered = cands.filter { it in 0..80 }.filterNot(::isConfirmed)
                if (filtered.isEmpty()) return null

                val ordered = filtered.sortedWith(
                    compareByDescending<Int> { conflictCountByIdx[it] }.thenBy { it }
                )
                val idx = ordered.first()
                val cc = conflictCountByIdx[idx]
                val cr = conflictReasonForIdx(idx)

                val reason = when {
                    cc > 0 && cr != null -> "$label + conflict($cc): $cr"
                    cc > 0 -> "$label + conflict($cc): involved in contradictions"
                    else -> "$label: needs verification on paper / scan uncertainty"
                }
                val pri = if (cc > 0) "conflict" else label
                return NextCheck(idx, pri, reason)
            }

            pickBest(gridState.mismatchCells, "mismatch")?.let { return it }

            if (gridState.solvability == "none") {
                pickBest(gridState.unresolvedCells, "unresolved")?.let { return it }
                return null
            }

            if (gridState.solvability == "unique") return null
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
                    appendLine("- If NOT currently pending confirm_validate: ask_user_to_confirm_validation (one clear yes/no).")
                    appendLine("- If pending confirm_validate and user reply is unclear: clarify_validation (do NOT repeat ask_user_to_confirm_validation).")
                    appendLine("- If pending confirm_validate and user confirmed match: finalize_validation_presentation + start_solving.")
                }

                gridState.solvability == "multiple" -> {
                    appendLine("- Ask user if the on-screen grid is a 100% match with the paper.")
                    appendLine("- If user confirms match but solvability remains multiple: recommend_retake.")
                    appendLine("- If user reply is unclear/garbled: clarify_validation (do NOT repeat ask_user_to_confirm_validation).")
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

            if (phase == GridPhase.SOLVING) {
                appendLine("ENGINE_STEP (authoritative; 1–3 lines):")
                if (engineStepSummary.isNullOrBlank()) {
                    appendLine("- (missing)")
                } else {
                    engineStepSummary
                        .replace("\r\n", "\n").replace('\r', '\n')
                        .lineSequence()
                        .map { it.trim() }
                        .filter { it.isNotBlank() }
                        .take(3)
                        .forEach { line -> appendLine("- $line") }
                }
                appendLine()
            } else {
                appendLine("CANDIDATES (USER THOUGHT PROCESS; MAY BE NOISY):")
                appendLine(candLines ?: "- (none)")
                appendLine()
            }

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
            if (phase != GridPhase.SOLVING) {
                appendLine("- candidate_marks_total: $totalCandidateMarks")
                appendLine("- candidate_marks_by_row (r1..r9): ${candByRow.toList()}")
                appendLine("- candidate_marks_by_col (c1..c9): ${candByCol.toList()}")
            }
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

            appendLine("VALIDATION_PROTOCOL_FLAGS (authoritative):")
            appendLine("- validation_mode: ${validationMode.name}    # GLOBAL|CELL_CHECK")
            appendLine("- pending_policy: ${pendingPolicy.name}      # CANCEL_ON_GLOBAL_VALIDATION|NONE")
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
            appendLine("- If validation_mode==GLOBAL: do NOT drive cell-by-cell verification; ask global match YES/NO.")
            appendLine("- For solvability==multiple: do NOT drive cell-by-cell; ask for 100% match; if match and still multiple => retake.")
            appendLine("- For solvability==unique and mismatch empty: stop corrections; offer grid as-is; ask if it matches paper.")
        }.trim()
    }
}