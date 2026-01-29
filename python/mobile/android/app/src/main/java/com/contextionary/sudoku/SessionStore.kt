package com.contextionary.sudoku

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

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
        Log.d(TAG, "setPending(): kind=$kind idx=$cellIdx digit=$digit")
    }

    @Synchronized
    fun clearPending() {
        step2State = step2State.copy(
            pendingKind = null,
            pendingCellIdx = null,
            pendingDigit = null
        )
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

        isAutoCorrected.set(false)
        isValidated.set(false)
        isReady.set(false)

        resetStep2Unsafe()

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
        val candidateMask: IntArray
    )

    @Synchronized
    fun truthSnapshotOrNull(): TruthSnapshot? {
        val g = truthIsGiven ?: return null
        val s = truthIsSolution ?: return null
        val c = truthCandidateMask ?: return null
        return TruthSnapshot(g.copyOf(), s.copyOf(), c.copyOf())
    }

    @Synchronized
    fun ensureTruthInitializedFromCanonicalIfPossible() {
        if (truthIsGiven != null && truthIsSolution != null && truthCandidateMask != null) return
        val g = canonicalIsGiven
        val s = canonicalIsSolution
        val c = canonicalCandidateMask
        if (g != null && s != null && c != null) {
            truthIsGiven = g.copyOf()
            truthIsSolution = s.copyOf()
            truthCandidateMask = c.copyOf()
            Log.d(TAG, "ensureTruthInitializedFromCanonicalIfPossible(): initialized from canonical")
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