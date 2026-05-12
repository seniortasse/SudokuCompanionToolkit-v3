package com.contextionary.sudoku

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

import com.contextionary.sudoku.conductor.policy.AssistantTallyV1
import com.contextionary.sudoku.conductor.policy.ContextSpanHintV1
import com.contextionary.sudoku.conductor.policy.RelationshipDeltaV1
import com.contextionary.sudoku.conductor.policy.RelationshipMemoryMergerV1
import com.contextionary.sudoku.conductor.policy.RelationshipMemoryV1
import com.contextionary.sudoku.conductor.policy.RepairSignalV1
import com.contextionary.sudoku.conductor.policy.TranscriptTurnV1
import com.contextionary.sudoku.conductor.policy.TurnContextV1
import com.contextionary.sudoku.conductor.policy.UserTallyV1



/**
 * SessionStore — single source of truth for the CURRENT capture session.
 *
 * Milestone M1 scope:
 * - Holds raw artifacts (run path, points, tiles) and raw 3-head cell readouts.
 * - Derives a simple M1 display board: prefer givenDigit, else solutionDigit, else 0.
 * - Exposes immutable snapshots for UI/consumers.
 * - Provides lifecycle flags: isReadyForValidation, isValidated.
 *
 * Step-2 (Conversation Confirmation) additions:
 * - Persist step2 phase: CONFIRMING vs LOCKED
 * - Persist mediationMode (UI gating for taps/picker)
 * - Persist pending cell (and pending kind/digit)
 */
object SessionStore {

    private const val TAG = "SessionStore"

    // -------------------------
    // Step-2 state (in-memory)
    // -------------------------

    enum class Step2Phase { CONFIRMING, LOCKED }

    enum class PendingKind {
        CONFIRM_CELL,        // “What’s in rXcY?”
        CONFIRM_EDIT,        // “Should I set rXcY to 3?”
        CONFIRM_RETAKE,      // “Want to retake?”
        CONFIRM_UNLOCK,      // “Do you want to unlock?”
        CONFIRM_SIGNOFF      // “Lock now?”
    }

    enum class RetakeRec { NONE, SOFT, STRONG }

    enum class Solvability { NONE, MULTIPLE, UNIQUE }

    data class Step2State(
        val phase: Step2Phase = Step2Phase.CONFIRMING,
        val mediationMode: Boolean = true,
        val pendingKind: PendingKind? = null,
        val pendingCellIdx: Int? = null,   // 0..80
        val pendingDigit: Int? = null,     // 1..9 (only when pendingKind==CONFIRM_EDIT)
        val lastRetakeRec: RetakeRec = RetakeRec.NONE,
        val lastSolvability: Solvability = Solvability.NONE
    )

    private var step2State: Step2State = Step2State()

    @Synchronized
    fun step2Snapshot(): Step2State = step2State

    @Synchronized
    fun setStep2Baseline(
        phase: Step2Phase = Step2Phase.CONFIRMING,
        mediationMode: Boolean = true,
        pendingKind: PendingKind? = null,
        pendingCellIdx: Int? = null,
        pendingDigit: Int? = null,
        lastRetakeRec: RetakeRec = RetakeRec.NONE,
        lastSolvability: Solvability = Solvability.NONE
    ) {
        step2State = Step2State(
            phase = phase,
            mediationMode = mediationMode,
            pendingKind = pendingKind,
            pendingCellIdx = pendingCellIdx,
            pendingDigit = pendingDigit,
            lastRetakeRec = lastRetakeRec,
            lastSolvability = lastSolvability
        )
        Log.d(TAG, "setStep2Baseline(): phase=$phase mediation=$mediationMode pending=$pendingKind idx=$pendingCellIdx digit=$pendingDigit retake=$lastRetakeRec solv=$lastSolvability")
    }

    @Synchronized
    fun setStep2Phase(phase: Step2Phase) {
        step2State = step2State.copy(phase = phase)
        Log.d(TAG, "setStep2Phase(): $phase")
    }

    @Synchronized
    fun setMediationMode(active: Boolean) {
        step2State = step2State.copy(mediationMode = active)
        Log.d(TAG, "setMediationMode(): $active")
    }

    @Synchronized
    fun setPending(
        kind: PendingKind?,
        cellIdx: Int? = null,
        digit: Int? = null
    ) {
        step2State = step2State.copy(
            pendingKind = kind,
            pendingCellIdx = cellIdx,
            pendingDigit = digit
        )

        // Phase 4: deterministic discourse update
        discourseTopicIssue = kind?.name?.lowercase()
        if (cellIdx != null) discourseTopicCell = cellNameOfIndex(cellIdx)

        discourseOpenAgendaIds.clear()
        if (kind != null) discourseOpenAgendaIds.add("pending:${kind.name.lowercase()}")

        Log.d(TAG, "setPending(): kind=$kind idx=$cellIdx digit=$digit")
    }

    @Synchronized
    fun clearPending() {
        step2State = step2State.copy(
            pendingKind = null,
            pendingCellIdx = null,
            pendingDigit = null
        )

        // Phase 4: deterministic discourse update
        discourseOpenAgendaIds.clear()
        discourseTopicIssue = null
        // keep discourseTopicCell (often still useful as “current topic”)

        Log.d(TAG, "clearPending()")
    }

    @Synchronized
    fun setLastRetakeRec(rec: RetakeRec) {
        step2State = step2State.copy(lastRetakeRec = rec)
    }

    @Synchronized
    fun setLastSolvability(sol: Solvability) {
        step2State = step2State.copy(lastSolvability = sol)
    }

    @Synchronized
    private fun resetStep2Unsafe() {
        step2State = Step2State()
    }



    // ---------------------------------------------------------------------
    // Phase 4: Discourse state (deterministic, App-owned)
    // ---------------------------------------------------------------------

    private var discourseTopicCell: String? = null          // e.g., "r4c2"
    private var discourseTopicIssue: String? = null         // e.g., "confirm_edit"|"ask_cell_value"|...
    private var discourseOpenAgendaIds: MutableList<String> = mutableListOf()

    private fun cellNameOfIndex(idx: Int): String {
        val i = idx.coerceIn(0, 80)
        val r = (i / 9) + 1
        val c = (i % 9) + 1
        return "r${r}c${c}"
    }

    @Synchronized
    fun snapshotDiscourseStateJson(): String {
        val o = JSONObject().apply {
            put("topic_cell", discourseTopicCell ?: "")
            put("topic_issue", discourseTopicIssue ?: "")
            put("open_agenda_ids", org.json.JSONArray().apply {
                discourseOpenAgendaIds.forEach { put(it) }
            })
        }
        return o.toString()
    }


    // ---------------------------------------------------------------------
    // Conversation memory (App brain owns 100%)
    // - per-session, cleared on reset()
    // ---------------------------------------------------------------------

    private var userTally: UserTallyV1 = UserTallyV1()
    private var assistantTally: AssistantTallyV1 = AssistantTallyV1.defaults()
    private var relationshipMemory: RelationshipMemoryV1 = RelationshipMemoryV1.defaults()

    // last 3 completed turns (user+assistant pairs)
    private val lastTurns: ArrayDeque<TranscriptTurnV1> = ArrayDeque()

    private var onboardingDone: Boolean = false

    @Synchronized fun getUserTally(): UserTallyV1 = userTally
    @Synchronized fun getAssistantTally(): AssistantTallyV1 = assistantTally
    @Synchronized fun getRelationshipMemory(): RelationshipMemoryV1 = relationshipMemory
    @Synchronized fun isOnboardingDone(): Boolean = onboardingDone
    @Synchronized fun setOnboardingDone(done: Boolean) { onboardingDone = done }

    @Synchronized
    fun setRelationshipMemory(memory: RelationshipMemoryV1) {
        relationshipMemory = memory
    }

    @Synchronized
    fun replaceRelationshipMemory(memory: RelationshipMemoryV1) {
        relationshipMemory = memory
    }

    @Synchronized
    fun resetRelationshipMemory() {
        relationshipMemory = RelationshipMemoryV1.defaults()
    }

    @Synchronized
    fun mergeRelationshipDelta(delta: RelationshipDeltaV1) {
        relationshipMemory = RelationshipMemoryMergerV1.merge(relationshipMemory, delta)
    }

    @Synchronized
    fun mergeUserTallyDelta(delta: UserTallyV1) {
        userTally = userTally.merge(delta)
    }

    @Synchronized
    fun mergeAssistantTallyDelta(delta: AssistantTallyV1) {
        assistantTally = assistantTally.merge(delta)
    }

    @Synchronized
    fun ensureFirstUserSpeech(firstSpeech: String) {
        if (userTally.firstSpeech.isNullOrBlank()) {
            userTally = userTally.merge(UserTallyV1(firstSpeech = firstSpeech))
        }
    }

    @Synchronized
    fun ensureFirstAssistantSpeech(firstSpeech: String) {
        if (assistantTally.firstSpeech.isNullOrBlank()) {
            assistantTally = assistantTally.merge(AssistantTallyV1(firstSpeech = firstSpeech))
        }
    }

    // Keep more than 3 turns so we can expand context when coreference is likely.
    // Phase 3: adaptive transcript window (3 → 6 → 10).
    private const val MAX_TRANSCRIPT_TURNS = 10

    @Synchronized
    fun appendTurnToTranscript(turnId: Long, userText: String, assistantText: String) {
        val t = TranscriptTurnV1(
            turnId = turnId,
            user = userText.trim(),
            assistant = assistantText.trim()
        )
        lastTurns.addLast(t)
        while (lastTurns.size > MAX_TRANSCRIPT_TURNS) lastTurns.removeFirst()
    }

    /** Returns the most recent N turns (oldest→newest), capped by what we have. */
    @Synchronized
    fun getRecentTurns(maxTurns: Int): List<TranscriptTurnV1> {
        val n = maxTurns.coerceIn(0, MAX_TRANSCRIPT_TURNS)
        val all = lastTurns.toList()
        return if (all.size <= n) all else all.takeLast(n)
    }

    /** Legacy convenience (kept): last 3 turns. */
    @Synchronized
    fun getLast3Turns(): List<TranscriptTurnV1> = getRecentTurns(3)

    /**
     * Series 6 — structured transcript lookback hints for Tick-1.
     *
     * Important:
     * This API must not infer meaning from raw user text.
     * Upstream callers should pass structured hints or state-derived pre-hints.
     */
    data class TranscriptContextHintsV1(
        val contextSpanHint: ContextSpanHintV1? = null,
        val referencesPriorTurns: Boolean? = null,
        val repairSignal: RepairSignalV1? = null,
        val hasOpenAgenda: Boolean = false
    )



    private fun isUserAgendaBridgeKeyV1(value: String?): Boolean =
        value?.contains("UserAgendaBridge", ignoreCase = true) == true

    /**
     * Series 6 — pre-Tick1 fallback hints derived only from authoritative state.
     *
     * Because this runs before the current Tick-1 parse exists, it may only use
     * already-known state such as pending/open agenda. It must not inspect raw user text.
     */
    fun derivePreTick1TranscriptContextHintsV1(
        pendingBefore: String?,
        lastAssistantQuestionKey: String?
    ): TranscriptContextHintsV1 {
        val isUserAgendaBridge =
            isUserAgendaBridgeKeyV1(pendingBefore) ||
                    isUserAgendaBridgeKeyV1(lastAssistantQuestionKey)

        val hasAppOpenAgenda =
            !isUserAgendaBridge &&
                    (!pendingBefore.isNullOrBlank() || !lastAssistantQuestionKey.isNullOrBlank())

        return TranscriptContextHintsV1(
            contextSpanHint =
                when {
                    isUserAgendaBridge -> ContextSpanHintV1.LOCAL
                    hasAppOpenAgenda -> ContextSpanHintV1.MEDIUM
                    else -> ContextSpanHintV1.LOCAL
                },
            referencesPriorTurns = hasAppOpenAgenda,
            repairSignal = null,
            hasOpenAgenda = hasAppOpenAgenda
        )
    }

    /**
     * Series 6 — adaptive transcript window selection for Tick-1 driven by structured hints.
     *
     * Policy:
     * - default 3
     * - medium 6 when nearby conversational context likely matters
     * - wide 10 for repair / explicit broader prior-turn dependence
     */
    @Synchronized
    fun getRecentTurnsAdaptive(
        hints: TranscriptContextHintsV1
    ): List<TranscriptTurnV1> {
        val n = when {
            hints.repairSignal == RepairSignalV1.MISHEARD ||
                    hints.repairSignal == RepairSignalV1.MISUNDERSTOOD ||
                    hints.repairSignal == RepairSignalV1.CONTRADICTION ||
                    hints.repairSignal == RepairSignalV1.LOOP_COMPLAINT ||
                    hints.repairSignal == RepairSignalV1.REPEATED_QUESTION -> 10

            hints.contextSpanHint == ContextSpanHintV1.LONG -> 10

            hints.contextSpanHint == ContextSpanHintV1.MEDIUM ||
                    hints.referencesPriorTurns == true ||
                    hints.hasOpenAgenda -> 6

            else -> 3
        }

        return getRecentTurns(n)
    }





    // ---------------------------------------------------------------------
// Conversation memory snapshot helper (for telemetry + LLM payloads)
// ---------------------------------------------------------------------

    data class MemorySnapshotJson(
        val userTallyJson: String,
        val assistantTallyJson: String,
        val relationshipMemoryJson: String,
        val recentTurnsJson: String
    )

    @Synchronized
    fun snapshotMemoryJson(): MemorySnapshotJson {
        // IMPORTANT:
        // Emit full-shape JSON so telemetry/audits always show all keys
        // even when the underlying fields are still null.
        val u = userTallyToFullJsonString(getUserTally())
        val a = assistantTallyToFullJsonString(getAssistantTally())
        val r = relationshipMemoryToFullJsonString(getRelationshipMemory())
        val t = TranscriptTurnV1.jsonArray(getLast3Turns()).toString()
        return MemorySnapshotJson(
            userTallyJson = u,
            assistantTallyJson = a,
            relationshipMemoryJson = r,
            recentTurnsJson = t
        )
    }

    private fun buildSolvingHandoffCtxV1(
        pendingBefore: String?,
        pendingTargetCell: String?,
        lastAssistantQuestionKey: String?,
        canonicalSolvingPositionKind: String?
    ): TurnContextV1.SolvingHandoffCtxV1? {
        val pendingUpper = pendingBefore?.trim()?.uppercase()
        val questionUpper = lastAssistantQuestionKey?.trim()?.uppercase()
        val canonicalUpper = canonicalSolvingPositionKind?.trim()?.uppercase()

        val isPostResolutionFollowup =
            canonicalUpper == "RESOLUTION_COMMIT" ||
                    canonicalUpper == "RESOLUTION_POST_COMMIT" ||
                    pendingUpper == "APPLY_HINT_NOW" ||
                    pendingUpper == "AFTER_RESOLUTION" ||
                    questionUpper == "APPLY_HINT_NOW" ||
                    questionUpper == "AFTER_RESOLUTION"

        if (!isPostResolutionFollowup) return null

        return TurnContextV1.SolvingHandoffCtxV1(
            handoffKind = "POST_RESOLUTION_CONTINUE",
            authority = "STRUCTURED_APP_STATE",
            commitAlreadyApplied = true,
            committedCell = pendingTargetCell,
            assistantCtaKind = "CONTINUE_ROUTE",
            assistantCtaScope = "NEXT_STEP",
            genericAssentDefaultIntent = "SOLVE_CONTINUE",
            detourOverrideRule = "ONLY_IF_EXPLICIT"
        )
    }

    @Synchronized
    fun snapshotTurnContextV1(
        turnId: Long,
        mode: String,
        phase: String,
        userText: String,
        pendingBefore: String?,
        pendingExpectedAnswerKind: String?,
        pendingTargetCell: String?,
        focusCell: String?,
        lastAssistantQuestionKey: String?,
        canonicalSolvingPositionKind: String? = null
    ): TurnContextV1 {
        val mem0 = snapshotMemoryJson()

        val isUserAgendaBridge =
            isUserAgendaBridgeKeyV1(pendingBefore) ||
                    isUserAgendaBridgeKeyV1(lastAssistantQuestionKey)

        // Series 6: structured transcript window hints for Tick-1.
        // Pre-Tick1, only authoritative prior state may be used here.
        val transcriptHints =
            derivePreTick1TranscriptContextHintsV1(
                pendingBefore = pendingBefore,
                lastAssistantQuestionKey = lastAssistantQuestionKey
            )

        val recentTurnsAdaptive = getRecentTurnsAdaptive(
            hints = transcriptHints
        )
        val recentTurnsJsonAdaptive = TranscriptTurnV1.jsonArray(recentTurnsAdaptive).toString()

        val awaitedAssistantAnswer =
            if (!pendingBefore.isNullOrBlank() || !lastAssistantQuestionKey.isNullOrBlank()) {
                TurnContextV1.AwaitedAssistantAnswerCtxV1(
                    owner = if (isUserAgendaBridge) "USER_AGENDA_OWNER" else "APP_ROUTE_OWNER",
                    questionKey = lastAssistantQuestionKey ?: pendingBefore,
                    questionKind = pendingBefore ?: lastAssistantQuestionKey,
                    expectedAnswerKind = pendingExpectedAnswerKind,
                    followupDisposition =
                        if (isUserAgendaBridge) "USER_AGENDA_DETOUR" else "APP_ROUTE_FOLLOWUP"
                )
            } else {
                null
            }

        val solvingHandoff =
            buildSolvingHandoffCtxV1(
                pendingBefore = pendingBefore,
                pendingTargetCell = pendingTargetCell,
                lastAssistantQuestionKey = lastAssistantQuestionKey,
                canonicalSolvingPositionKind = canonicalSolvingPositionKind
            )

        return TurnContextV1(
            turnId = turnId,
            mode = mode,
            phase = phase,
            userText = userText,
            pending = TurnContextV1.PendingCtxV1(
                pendingBefore = pendingBefore,
                expectedAnswerKind = pendingExpectedAnswerKind,
                targetCell = pendingTargetCell
            ),
            focusCell = focusCell,
            focusCoreferencePolicy =
                if (isUserAgendaBridge) "STRICT_DEICTIC_ONLY" else "NORMAL",
            lastAssistantQuestionKey = lastAssistantQuestionKey,
            canonicalSolvingPositionKind = canonicalSolvingPositionKind,
            solvingHandoff = solvingHandoff,
            awaitedAssistantAnswer = awaitedAssistantAnswer,
            recentTurnsPolicy =
                if (isUserAgendaBridge) "DEICTIC_ONLY_NO_CELL_COMPLETION" else "ADAPTIVE",
            userTallyJson = mem0.userTallyJson,
            assistantTallyJson = mem0.assistantTallyJson,
            recentTurnsJson = recentTurnsJsonAdaptive
        )
    }

    // -------- Raw artifacts from rectification / capture
    private var rectifyRunPath: String? = null
    private var points10x10: FloatArray? = null          // Optional (from points_10x10.json)
    private var tilePaths: List<String>? = null          // Optional (81 cropped tiles on disk)

    // -------- Raw 3-head per-cell readouts (length must be 81)
    private var cellReadouts: Array<CellReadout>? = null

    // -------- Simple M1 display board (derived)
    // Prefer givenDigit, else solutionDigit, else 0
    private var digitsForDisplay: IntArray? = null
    private var avgGivenConf: Float = 0f
    private var avgSolutionConf: Float = 0f
    private var nonZeroDigitsCount: Int = 0

    // -------- Canonical post-autocorrect board (optional)
    // This is what downstream stages should use once populated.
    private var canonicalDigits: IntArray? = null
    private var canonicalConfs: FloatArray? = null
    private var canonicalIsGiven: BooleanArray? = null
    private var canonicalIsSolution: BooleanArray? = null
    private var canonicalCandidateMask: IntArray? = null
    private var canonicalChangedIndices: IntArray? = null
    private var canonicalUnresolvedIndices: IntArray? = null




    // -------- Truth provenance (user-overridable)
    private var truthIsGiven: BooleanArray? = null
    private var truthIsSolution: BooleanArray? = null

    // Candidates are inherently user-thought; we store them as "truth candidates"
    private var truthCandidateMask: IntArray? = null

    private var truthConfirmed: BooleanArray? = null



    // -------- Lifecycle flags
    private val isReady = AtomicBoolean(false)
    private val isValidated = AtomicBoolean(false)
    private val isAutoCorrected = AtomicBoolean(false)

    /**
     * Clears the current session completely.
     * Call this on "Retake" or when returning to Camera screen.
     */
    @Synchronized
    fun reset() {


        userTally = UserTallyV1()
        assistantTally = AssistantTallyV1.defaults()
        relationshipMemory = RelationshipMemoryV1.defaults()
        lastTurns.clear()
        onboardingDone = false

        rectifyRunPath = null
        points10x10 = null
        tilePaths = null
        cellReadouts = null

        digitsForDisplay = null
        avgGivenConf = 0f
        avgSolutionConf = 0f
        nonZeroDigitsCount = 0

        canonicalDigits = null
        canonicalConfs = null
        canonicalIsGiven = null
        canonicalIsSolution = null
        canonicalCandidateMask = null
        canonicalChangedIndices = null
        canonicalUnresolvedIndices = null

        truthIsGiven = null
        truthIsSolution = null
        truthCandidateMask = null
        truthConfirmed = null

        isAutoCorrected.set(false)
        isValidated.set(false)
        isReady.set(false)

        resetStep2Unsafe()

        truthConfirmed = BooleanArray(81)

        Log.d(TAG, "reset() → session cleared (including Step-2 state)")
    }

    /**
     * Ingests the capture results (rectify run + 3-head outputs) and computes a simple M1 board.
     *
     * @return an immutable SessionSnapshot reflecting the new state
     */
    @Synchronized
    fun ingestCapture(
        runPath: String,
        readouts: Array<CellReadout>,
        p10x10: FloatArray? = null,
        tiles: List<String>? = null
    ): SessionSnapshot {
        require(readouts.size == 81) {
            "SessionStore.ingestCapture requires exactly 81 CellReadout items"
        }

        rectifyRunPath = runPath
        points10x10 = p10x10
        tilePaths = tiles
        cellReadouts = readouts

        // Clear any previous autocorrect results (new capture supersedes them)
        canonicalDigits = null
        canonicalConfs = null
        canonicalIsGiven = null
        canonicalIsSolution = null
        canonicalCandidateMask = null
        canonicalChangedIndices = null
        canonicalUnresolvedIndices = null
        isAutoCorrected.set(false)

        // New capture => Step-2 state resets to default
        resetStep2Unsafe()

        // M1 "display digits": prefer given → solution → 0
        val display = IntArray(81)
        var gConfSum = 0f
        var sConfSum = 0f
        var nz = 0

        for (i in 0 until 81) {
            val r = readouts[i]
            gConfSum += r.givenConf
            sConfSum += r.solutionConf
            val d = when {
                r.givenDigit in 1..9 -> r.givenDigit
                r.solutionDigit in 1..9 -> r.solutionDigit
                else -> 0
            }
            if (d != 0) nz++
            display[i] = d
        }

        digitsForDisplay = display
        avgGivenConf = gConfSum / 81f
        avgSolutionConf = sConfSum / 81f
        nonZeroDigitsCount = nz

        isValidated.set(false)
        isReady.set(true)

        Log.d(
            TAG,
            "ingestCapture(): run='$runPath', nonZero=$nz, " +
                    "avgGivenConf=${"%.3f".format(avgGivenConf)}, " +
                    "avgSolutionConf=${"%.3f".format(avgSolutionConf)}"
        )

        return snapshotUnsafe()
    }


    data class TruthSnapshot(
        val isGiven: BooleanArray,
        val isSolution: BooleanArray,
        val candidateMask: IntArray,
        val confirmed81: BooleanArray
    )

    @Synchronized
    fun truthSnapshotOrNull(): TruthSnapshot? {
        val g = truthIsGiven ?: return null
        val s = truthIsSolution ?: return null
        val c = truthCandidateMask ?: return null
        val k = truthConfirmed ?: BooleanArray(81)
        return TruthSnapshot(g.copyOf(), s.copyOf(), c.copyOf(), k.copyOf())
    }

    @Synchronized
    fun ensureTruthInitializedFromCanonicalIfPossible() {
        // If baseline truth exists, still ensure confirmed exists.
        if (truthIsGiven != null && truthIsSolution != null && truthCandidateMask != null) {
            if (truthConfirmed == null || truthConfirmed?.size != 81) truthConfirmed = BooleanArray(81)
            return
        }

        val g = canonicalIsGiven
        val s = canonicalIsSolution
        val c = canonicalCandidateMask
        if (g != null && s != null && c != null) {
            truthIsGiven = g.copyOf()
            truthIsSolution = s.copyOf()
            truthCandidateMask = c.copyOf()
            if (truthConfirmed == null || truthConfirmed?.size != 81) truthConfirmed = BooleanArray(81)
            Log.d(TAG, "ensureTruthInitializedFromCanonicalIfPossible(): initialized from canonical (+confirmed81)")
        }
    }

    @Synchronized
    fun reclassifyCell(idx: Int, kind: String): Boolean {
        ensureTruthInitializedFromCanonicalIfPossible()
        val g = truthIsGiven ?: return false
        val s = truthIsSolution ?: return false
        if (idx !in 0..80) return false

        when (kind.trim().lowercase()) {
            "given" -> { g[idx] = true;  s[idx] = false }
            "solution" -> { g[idx] = false; s[idx] = true }
            "neither" -> { g[idx] = false; s[idx] = false }
            else -> return false
        }
        return true
    }

    @Synchronized
    fun setConfirmed(idx: Int, confirmed: Boolean): Boolean {
        ensureTruthInitializedFromCanonicalIfPossible()
        val k = truthConfirmed ?: run {
            truthConfirmed = BooleanArray(81)
            truthConfirmed!!
        }
        if (idx !in 0..80) return false
        k[idx] = confirmed
        return true
    }

    @Synchronized
    fun clearAllConfirmed() {
        truthConfirmed = BooleanArray(81)
    }

    @Synchronized
    fun setCandidates(idx: Int, mask: Int): Boolean {
        ensureTruthInitializedFromCanonicalIfPossible()
        val c = truthCandidateMask ?: return false
        if (idx !in 0..80) return false
        if (mask !in 0..0x1FF) return false
        c[idx] = mask
        return true
    }

    @Synchronized
    fun toggleCandidate(idx: Int, digit: Int): Boolean {
        ensureTruthInitializedFromCanonicalIfPossible()
        val c = truthCandidateMask ?: return false
        if (idx !in 0..80) return false
        if (digit !in 1..9) return false
        val bit = 1 shl (digit - 1)
        c[idx] = c[idx] xor bit
        return true
    }



    /**
     * Ingests the post-autocorrect *canonical* board (digits + typing + candidates).
     */
    @Synchronized
    fun ingestAutocorrect(
        digits81: IntArray,
        confidences81: FloatArray,
        isGiven81: BooleanArray,
        isSolution81: BooleanArray,
        candidateMask81: IntArray,
        changedIndices: IntArray,
        unresolvedIndices: IntArray
    ): SessionSnapshot {
        check(isReady.get()) { "ingestAutocorrect() called before ingestCapture(); not ready" }

        require(digits81.size == 81) { "digits81 must have size 81" }
        require(confidences81.size == 81) { "confidences81 must have size 81" }
        require(isGiven81.size == 81) { "isGiven81 must have size 81" }
        require(isSolution81.size == 81) { "isSolution81 must have size 81" }
        require(candidateMask81.size == 81) { "candidateMask81 must have size 81" }

        canonicalDigits = digits81.copyOf()
        canonicalConfs = confidences81.copyOf()
        canonicalIsGiven = isGiven81.copyOf()
        canonicalIsSolution = isSolution81.copyOf()
        canonicalCandidateMask = candidateMask81.copyOf()
        canonicalChangedIndices = changedIndices.copyOf()
        canonicalUnresolvedIndices = unresolvedIndices.copyOf()

        // Initialize truth provenance from canonical (first capture baseline)
        truthIsGiven = canonicalIsGiven?.copyOf()
        truthIsSolution = canonicalIsSolution?.copyOf()
        truthCandidateMask = canonicalCandidateMask?.copyOf()
        truthConfirmed = truthConfirmed?.takeIf { it.size == 81 } ?: BooleanArray(81)

        isAutoCorrected.set(true)

        // Also update the "digitsForDisplay" so older UI paths still show corrected digits.
        digitsForDisplay = canonicalDigits

        val nz = canonicalDigits!!.count { it != 0 }
        nonZeroDigitsCount = nz

        Log.d(
            TAG,
            "ingestAutocorrect(): nonZero=$nz changed=${changedIndices.size} unresolved=${unresolvedIndices.size}"
        )

        return snapshotUnsafe()
    }

    /**
     * Immutable snapshot for UI/consumers.
     * Safe to call only when isReadyForValidation == true.
     */
    @Synchronized
    fun snapshot(): SessionSnapshot {
        check(isReady.get()) {
            "SessionStore.snapshot() called before ingest; not ready"
        }
        return snapshotUnsafe()
    }

    /**
     * Marks the current grid as validated (after user presses Keep).
     */
    fun markValidated() {
        isValidated.set(true)
        Log.d(TAG, "markValidated() → isValidated=true")
    }

    /** True once ingestCapture() has been called successfully. */
    fun isReadyForValidation(): Boolean = isReady.get()

    /** True after markValidated() is called in this session. */
    fun isValidated(): Boolean = isValidated.get()

    /** True once ingestAutocorrect() has been called for the current capture. */
    fun isAutoCorrected(): Boolean = isAutoCorrected.get()

    // -------- Internal

    @Synchronized
    private fun snapshotUnsafe(): SessionSnapshot {
        val display = digitsForDisplay
            ?: error("digitsForDisplay is null despite isReady=true")

        val s2 = step2State

        return SessionSnapshot(
            rectifyRunPath = rectifyRunPath,
            points10x10 = points10x10,
            tilePaths = tilePaths,
            cellReadouts = cellReadouts!!,
            digitsForDisplay = display,
            avgGivenConf = avgGivenConf,
            avgSolutionConf = avgSolutionConf,
            nonZeroDigitsCount = nonZeroDigitsCount,
            isReadyForValidation = isReady.get(),
            isValidated = isValidated.get(),

            // ---- appended fields (safe defaults in data class)
            canonicalDigits = canonicalDigits,
            canonicalConfidences = canonicalConfs,
            canonicalIsGiven = canonicalIsGiven,
            canonicalIsSolution = canonicalIsSolution,
            canonicalCandidateMask = canonicalCandidateMask,
            changedIndices = canonicalChangedIndices,
            unresolvedIndices = canonicalUnresolvedIndices,
            isAutoCorrected = isAutoCorrected.get(),

            // ---- Step-2 persisted (in-memory)
            step2Phase = s2.phase.name,
            step2MediationMode = s2.mediationMode,
            step2PendingKind = s2.pendingKind?.name,
            step2PendingCellIdx = s2.pendingCellIdx,
            step2PendingDigit = s2.pendingDigit
        )
    }



    // ---------------------------------------------------------------------
    // Full-shape JSON (for telemetry + LLM context)
    // - Always includes ALL keys, using "" for missing values.
    // - Prevents “only first_speech shows up” when other fields are null.
    // ---------------------------------------------------------------------

    private fun userTallyToFullJsonString(u: UserTallyV1): String {
        val o = org.json.JSONObject()
        o.put("name", u.name ?: "")
        o.put("age", u.age ?: "")
        o.put("facts", u.facts ?: "")
        o.put("sudoku_level", u.sudokuLevel ?: "")
        o.put("thinking_process", u.thinkingProcess ?: "")
        o.put("dislikes", u.dislikes ?: "")
        o.put("preferences", u.preferences ?: "")
        o.put("personality", u.personality ?: "")
        o.put("first_speech", u.firstSpeech ?: "")
        return o.toString()
    }

    private fun assistantTallyToFullJsonString(a: AssistantTallyV1): String {
        val o = org.json.JSONObject()
        o.put("name", a.name ?: "")
        o.put("age", a.age ?: "")
        o.put("about", a.about ?: "")
        o.put("dislikes", a.dislikes ?: "")
        o.put("preferences", a.preferences ?: "")
        o.put("personality", a.personality ?: "")
        o.put("first_speech", a.firstSpeech ?: "")
        return o.toString()
    }

    private fun relationshipMemoryToFullJsonString(r: RelationshipMemoryV1): String {
        return r.toJson().toString()
    }


}

/**
 * Immutable view of the current session.
 * NOTE: This is an in-memory snapshot; it is not persisted across app restarts.
 */
data class SessionSnapshot(
    val rectifyRunPath: String?,
    val points10x10: FloatArray?,
    val tilePaths: List<String>?,
    val cellReadouts: Array<CellReadout>,
    val digitsForDisplay: IntArray,
    val avgGivenConf: Float,
    val avgSolutionConf: Float,
    val nonZeroDigitsCount: Int,
    val isReadyForValidation: Boolean,
    val isValidated: Boolean,

    // ---- M2+ (optional) canonical post-autocorrect board
    val canonicalDigits: IntArray? = null,
    val canonicalConfidences: FloatArray? = null,
    val canonicalIsGiven: BooleanArray? = null,
    val canonicalIsSolution: BooleanArray? = null,
    val canonicalCandidateMask: IntArray? = null,
    val changedIndices: IntArray? = null,
    val unresolvedIndices: IntArray? = null,
    val isAutoCorrected: Boolean = false,

    // ---- Step-2 persisted (in-memory)
    val step2Phase: String = "CONFIRMING",
    val step2MediationMode: Boolean = true,
    val step2PendingKind: String? = null,
    val step2PendingCellIdx: Int? = null,
    val step2PendingDigit: Int? = null


)