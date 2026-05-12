package com.contextionary.sudoku.conductor

import com.contextionary.sudoku.logic.LLMGridState
import com.contextionary.sudoku.conductor.solving.CoachPlan
import com.contextionary.sudoku.conductor.solving.DetourNarrativeContextV1
import com.contextionary.sudoku.conductor.policy.*

enum class SudoMode { GRID_SESSION, FREE_TALK }

enum class GridPhase { CONFIRMING, SEALING, SOLVING }

enum class TurnOwnerV1 {
    USER_OWNER,
    APP_OWNER,
    NONE
}

/**
 * Wave 1 — constitutional turn authority.
 *
 * TurnOwnerV1 remains the legacy coarse lane owner used by existing code paths.
 * TurnAuthorityOwnerV1 is the new sovereign owner model that later waves will
 * make authoritative for demand selection, projection, pending suspension,
 * route jumps, and repair handling.
 */
enum class TurnAuthorityOwnerV1 {
    APP_ROUTE_OWNER,
    USER_DETOUR_OWNER,
    USER_ROUTE_JUMP_OWNER,
    REPAIR_OWNER,
    NONE;

    fun toLegacyTurnOwnerV1(): TurnOwnerV1 =
        when (this) {
            APP_ROUTE_OWNER -> TurnOwnerV1.APP_OWNER
            USER_DETOUR_OWNER,
            USER_ROUTE_JUMP_OWNER,
            REPAIR_OWNER -> TurnOwnerV1.USER_OWNER
            NONE -> TurnOwnerV1.NONE
        }
}

/**
 * Wave 1 — normalized evidence buckets for the sovereign traffic controller.
 *
 * These are not yet used to arbitrate the turn in Wave 1; they exist so later
 * waves can migrate existing signals into one constitutional decision object
 * instead of letting several independent mini-authorities compete.
 */
enum class TurnAuthorityEvidenceKindV1 {
    APP_QUEUE,
    USER_QUEUE,
    INTENT_ROUTE_FOLLOW,
    INTENT_DETOUR,
    INTENT_ROUTE_JUMP,
    INTENT_REPAIR,
    CANONICAL_POSITION,
    PENDING_STATE,
    SUPPORT_PACKET,
    ROUTE_CHECKPOINT,
    DETOUR_STATE,
    NONE
}

data class TurnAuthorityEvidenceV1(
    val kind: TurnAuthorityEvidenceKindV1,
    val detail: String? = null
)

/**
 * Wave 1 — one structured record for "who owns this turn and why".
 *
 * Later waves will populate this from the arbiter and make all downstream
 * demand/projection logic subordinate to it.
 */
enum class PendingAuthorityStatusV1 {
    ACTIVE,
    SUSPENDED,
    TRANSFORMED,
    SUBORDINATED,
    CLEARED,
    NONE
}

enum class RouteBoundaryStatusV1 {
    STAY_ON_CURRENT_CHECKPOINT,
    PAUSED_AT_CHECKPOINT,
    REENTER_TARGET_CHECKPOINT,
    RELEASED_TO_NEXT_STEP,
    BLOCKED_BY_REPAIR,
    NONE
}

data class RouteBoundaryDecisionV1(
    val status: RouteBoundaryStatusV1 = RouteBoundaryStatusV1.NONE,
    val reason: String = "uninitialized",
    val fromCheckpointKind: String? = null,
    val toCheckpointKind: String? = null,
    val routePaused: Boolean = false,
    val advanceAllowed: Boolean = false
)

data class TurnAuthorityDecisionV1(
    val owner: TurnAuthorityOwnerV1 = TurnAuthorityOwnerV1.NONE,
    val reason: String = "uninitialized",
    val evidence: List<TurnAuthorityEvidenceV1> = emptyList(),
    val canonicalPositionKind: String? = null,
    val pendingKind: String? = null,
    val appQueueHeadKind: String? = null,
    val userQueueHeadKind: String? = null,
    val routeJumpRequested: Boolean = false,
    val repairRequested: Boolean = false,
    val detourRequested: Boolean = false,
    val appRouteRequested: Boolean = false,
    val suspendedPendingKind: String? = null,
    val pendingAuthorityStatus: PendingAuthorityStatusV1 = PendingAuthorityStatusV1.NONE,
    val resumedPendingKind: String? = null,
    val routeReturnAllowed: Boolean = false
)

enum class OverlayOwnerV1 {
    USER_OWNER,
    APP_OWNER,
    CHECKPOINT_RESTORE,
    NONE
}

enum class OverlayContractKindV1 {
    PRESERVE_CURRENT,
    RESTORE_CHECKPOINT,
    SHOW_STORY_SCOPE,
    SPOTLIGHT_CELL,
    SHOW_ATOM_FRAME,
    CLEAR
}

data class OverlayContractV1(
    val kind: OverlayContractKindV1,
    val frameIds: List<String> = emptyList(),
    val focusCellRef: String? = null,
    val reason: String? = null
)


// ------------------------------------------------------------
// Step 6 — deterministic solve-step compute latch ("compute then speak")
// ------------------------------------------------------------
enum class SolveStepStatus { IDLE, COMPUTING, READY, FAILED }

// ------------------------------------------------------------
// Phase 1 — Narrative Story (3-act step narration; first-class state)
// ------------------------------------------------------------
enum class StoryStage { SETUP, CONFRONTATION, RESOLUTION }

// ------------------------------------------------------------
// Phase 2 — Step-level guidance mode (sticky within the current step)
// ------------------------------------------------------------
enum class SolveGuidanceModeV1 { UNSET, GUIDE, TRY_SELF }

/**
 * Deterministic story context for the current solve step.
 * - Stored in state so detours/clarifications can resume the same story stage
 * - Atom 0 ("SPOTLIGHT") is the setup anchor and should remain visible throughout the step.
 */
data class NarrativeStoryContext(
    val stepId: String,
    val gridHash12: String,
    val stage: StoryStage,
    val atomsCount: Int,

    /**
     * Phase 8: proof beats are atoms 1..lastProofAtomIndex, commit is atom (atomsCount-1).
     * If atomsCount < 2, lastProofAtomIndex becomes 0 (no proof beats).
     */
    val lastProofAtomIndex: Int = ((atomsCount - 2).coerceAtLeast(0)),

    val focusAtomIndex: Int? = 0,                 // anchor atom for the stage: 0 in SETUP; proof anchor in CONFRONTATION; commit/reveal atom in RESOLUTION
    val discussedAtomIndices: List<Int> = emptyList(), // narration atoms explicitly discussed in this stage (not necessarily the full overlay/context scope)
    val readyForCommit: Boolean = false,
    val createdTurnSeq: Long
)



enum class SolveStepRequestKindV1 {
    PREFETCH_CURRENT_GRID,
    NEXT_STEP_CONTINUATION,
    GRID_REFRESH,
    SOLVING_ENTRY
}

data class PendingSolveStepAutocoach(
    val reason: String,
    val gridHash12: String,
    val origin: String,      // e.g. "solving_entry" | "user_requested_next_step" | "refresh"
    val turnSeq: Long,

    // Batch C:
    // structured request metadata so next-step admission does not depend
    // on a magic reason string.
    val requestKind: SolveStepRequestKindV1 = SolveStepRequestKindV1.PREFETCH_CURRENT_GRID,
    val requestToken: String? = null,
    val allowReplaceLockedStep: Boolean = false
)



// ------------------------------------------------------------
// Phase 0 — Route context contract (scaffold; wired in Phase 1+)
// ------------------------------------------------------------
data class ActiveRouteContextV2(
    val gridHash12: String,
    val stepId: String? = null,          // engine step id if available
    val stepHash12: String? = null,      // hash of step packet if available
    val techniqueId: String? = null,

    // NOTE: CellRefV1 is not defined in this project build.
    // Keep this as a simple rXcY string if you need it later.
    val targetCell: String? = null,

    val createdAtMs: Long = System.currentTimeMillis(),
    val routeVersion: Int = 1            // bump on recompute
)

// ------------------------------------------------------------
// Phase 2 — Persisted route checkpoint
// Exact paused place on the app-owned route.
// ------------------------------------------------------------
data class RouteCheckpointV1(
    val routeId: String,
    val phase: GridPhase,
    val appAgendaItemId: String? = null,
    val appAgendaKind: String? = null,
    val stepId: String? = null,
    val storyStage: StoryStage? = null,
    val focusCellIndex: Int? = null,
    val targetCellIndex: Int? = null,
    val targetDigit: Int? = null,
    val techniqueId: String? = null,
    val overlayFrameIds: List<String> = emptyList(),
    val replyDemandKind: String? = null,
    val resumePromptHint: String? = null,
    val canonicalPositionKind: CanonicalSolvingPositionKindV1? = null,
    val canonicalHeadKind: CanonicalSolvingHeadKindV1? = null,
    val canonicalPendingKind: CanonicalSolvingPendingKindV1? = null,
    val gridHash12: String? = null,
    val focusAtomIndex: Int? = null,
    val atomsCount: Int? = null,
    val readyForCommit: Boolean = false,
    val createdTurnSeq: Long,
    val routeVersion: Int = 1
)

// ------------------------------------------------------------
// Phase 0 — Detour state scaffold (no routing behavior yet)
// ------------------------------------------------------------
enum class DetourAnswerStateV1 {
    ACTIVE_UNANSWERED,
    ANSWERED_READY_TO_RETURN,
    FAILED_AWAITING_RECOVERY
}

enum class DetourFailureRecoveryDecisionV1 {
    NONE,
    RETRY_DETOUR,
    RETIRE_AND_RESUME
}

data class DetourTurnRecordV1(
    val turnSeq: Long,
    val questionClass: DetourQuestionClassV1? = null,
    val agendaId: String? = null,
    val agendaKind: String? = null,
    val requestedCell: String? = null,
    val requestedDigit: Int? = null,
    val requestedSearchScope: String? = null,
    val answerState: DetourAnswerStateV1 = DetourAnswerStateV1.ACTIVE_UNANSWERED,
    val failureReason: String? = null,
    val anchorStepId: String? = null,
    val anchorStage: StoryStage? = null
)

enum class DetourSemanticSourceV1 {
    NONE,
    CURRENT_TURN_DETOUR_STATE
}

data class DetourStateV1(
    val isActive: Boolean = false,
    val questionClass: DetourQuestionClassV1? = null,

    val semanticSource: DetourSemanticSourceV1 = DetourSemanticSourceV1.NONE,
    val activeAgendaId: String? = null,
    val activeAgendaKind: String? = null,

    // Materialized current-turn detour semantics.
    val proofAskKind: ProofChallengeAskKindV1? = null,
    val proofLaneHint: DetourProofChallengeLaneV1? = null,
    val proofRivalCellRef: String? = null,
    val proofRivalDigit: Int? = null,
    val proofClaimedTechniqueId: String? = null,
    val proofObjectHint: String? = null,

    val targetAskKind: TargetCellAskKindV1? = null,
    val candidateAskKind: CandidateAskKindV1? = null,
    val neighborAskKind: NeighborAskKindV1? = null,

    // Track A / Phase A1 — explicit detour answer state.
    val answerState: DetourAnswerStateV1 = DetourAnswerStateV1.ACTIVE_UNANSWERED,
    val answerFailureReason: String? = null,
    val answeredTurnSeq: Long? = null,
    val failedTurnSeq: Long? = null,

    // Track A / Phase A2 — explicit recovery policy after failed detour answers.
    val failureRecoveryDecision: DetourFailureRecoveryDecisionV1 = DetourFailureRecoveryDecisionV1.NONE,
    val failureRecoveryDecisionTurnSeq: Long? = null,

    // Anchor back to the paused solving route.
    val anchorStepId: String? = null,
    val anchorStage: StoryStage? = null,
    val anchorTargetCell: String? = null,
    val anchorTechniqueId: String? = null,

    // Optional local references explicitly mentioned by the user.
    val requestedCell: String? = null,
    val requestedDigit: Int? = null,
    val requestedTechniqueIdOrLabel: String? = null,
    val requestedSearchScope: String? = null, // "local" | "global" | null

    // Pending solver-backed detour query (SV-3 live integration).
    val pendingSolverQueryId: String? = null,
    val pendingSolverQueryOp: String? = null,

    // Overlay preservation / override.
    val pausedStoryAppliedFrameIds: List<String> = emptyList(),
    val detourAppliedFrameIds: List<String> = emptyList(),

    // Cheap control / telemetry fields.
    val detourTurnCount: Int = 0,
    val returnHint: String? = null
)


// ------------------------------------------------------------
// Phase 0 — Turn Contract Snapshot (runtime-checked invariants)
// ------------------------------------------------------------
data class TurnContractSnapshotV1(
    val stage: String,                 // e.g. "POST_DECISION" | "POST_APPLY"
    val turnSeq: Long,
    val mode: String,
    val phase: String,

    val pendingKind: String?,
    val agendaItemKind: String?,

    val ctaArtifactsPresent: List<String>,
    val ctaArtifactsCount: Int,

    val rawTextUsedByMaps: Boolean,

    val proofBundlesPresent: List<String>,
    val proofOkForSolving: Boolean
)






// ------------------------------------------------------------
// Phase 1 — Pending-driven CTA choice ids (North Star doctrine)
// ------------------------------------------------------------
object PendingChoiceId {
    // --------------------------------------------------------
    // Canonical North Star rail
    // --------------------------------------------------------
    const val SHOW_PROOF = "show_proof"     // setup -> guided proof
    const val LOCK_IT_IN = "lock_it_in"     // confrontation -> commit/apply
    const val NEXT_STEP = "next_step"       // resolution -> continue solving

    // --------------------------------------------------------
    // Explicit detours / repair / off-rail branches
    // These are NOT normal sibling CTAs in the North Star rail.
    // They remain for transitional compatibility and will be
    // handled more explicitly in later cleanup phases.
    // --------------------------------------------------------
    const val TRY_SELF = "try_self"
    const val EXPLAIN_MORE = "explain_more"
    const val ASK_TECHNIQUE = "ask_technique"
    const val RETURN_TO_ROUTE = "return_to_route"

    // --------------------------------------------------------
    // Temporary legacy aliases (keep compile stability while
    // downstream conductor/binder code is still being purified)
    // --------------------------------------------------------
    const val GUIDE_ME = SHOW_PROOF
    const val READY_FOR_ANSWER = LOCK_IT_IN
    const val APPLY_NOW = LOCK_IT_IN
    const val LET_ME_TRY = TRY_SELF
}








sealed class Pending {

    // Existing
    data class ConfirmEdit(
        val cellIndex: Int,          // 0..80
        val proposedDigit: Int,       // 1..9
        val source: String,           // "llm" | "user_tap" | etc
        val prompt: String            // what we asked the user
    ) : Pending()

    // Wave-3 pending-state normalization:
    // make bounded transactional contracts first-class in the conductor state model.
    data class ConfirmCellAsIs(
        val cellIndex: Int,
        val prompt: String,
        val source: String = "system"
    ) : Pending()

    data class ConfirmCellToDigit(
        val cellIndex: Int,
        val proposedDigit: Int,
        val prompt: String,
        val source: String = "system"
    ) : Pending()

    data class ConfirmRegionAsIs(
        val regionKind: String,       // "row" | "column" | "box"
        val regionIndex1Based: Int,   // 1..9
        val prompt: String,
        val source: String = "system"
    ) : Pending()

    data class ConfirmRegionToDigits(
        val regionKind: String,       // "row" | "column" | "box"
        val regionIndex1Based: Int,   // 1..9
        val digits: List<Int>,
        val prompt: String,
        val source: String = "system"
    ) : Pending()

    data class ProvideDigit(
        val cellIndex: Int? = null,
        val row: Int? = null,
        val col: Int? = null,
        val prompt: String
    ) : Pending()

    data class AskCellValue(
        val cellIndex: Int,
        val prompt: String
    ) : Pending()

    data class ConfirmRetake(
        val strength: String,         // "soft"|"strong"
        val prompt: String
    ) : Pending()

    data class ConfirmValidate(
        val prompt: String
    ) : Pending()

    /**
     * AGENDA 6: Grid is uniquely solvable; we already finalized presentation.
     * Now ask whether to enter SOLVING mode.
     */
    data class ConfirmStartSolving(
        val prompt: String
    ) : Pending()

    /**
     * AGENDA 4: Visual verification — user confirms 100% match between on-screen and paper.
     * YES => set userConfirmedPerfectMatch=true and continue agenda.
     * NO  => ask for the list of cells to edit (clarification becomes the CTA).
     */
    data class VisualVerifyMatch(
        val prompt: String
    ) : Pending()

    // ----------------------------
    // Gate 4: repair conversation states
    // ----------------------------

    data class ConfirmInterpretation(
        val row: Int? = null,         // 1..9
        val col: Int? = null,         // 1..9
        val digit: Int? = null,       // 1..9
        val confidence: Float = 0.6f,
        val prompt: String
    ) : Pending()

    data class AskClarification(
        val kind: ClarifyKind,
        val rowHint: Int? = null,
        val colHint: Int? = null,
        val digitHint: Int? = null,
        val prompt: String
    ) : Pending()

    data class WaitForTap(
        val prompt: String,
        val digitHint: Int? = null,
        val confidence: Float = 0.6f
    ) : Pending()

    // ----------------------------
    // SOLVING: recovery pending (Patch 2)
    // ----------------------------

    /**
     * Solver step failed/missing. We do NOT auto-call LLM.
     * We wait for explicit user action ("next step") or a tap request.
     */
    data class SolveStepMissing(
        val prompt: String,
        val gridHash12: String? = null,

        // M1 fallback-commit path:
        // the solver move is still trusted, but the proof packet failed.
        // When autoApplyFallback is true, Tick2 should close the turn honestly,
        // name the move, and the conductor should apply the digit in the same turn.
        val fallbackStepId: String? = null,
        val fallbackCellIndex: Int? = null,
        val fallbackDigit: Int? = null,
        val fallbackTechnique: String? = null,
        val fallbackFailureReason: String? = null,
        val autoApplyFallback: Boolean = false
    ) : Pending()

    // ----------------------------
    // SOLVING: end-of-turn CTA pending
    // ----------------------------

    /**
     * Phase 6: stage-native solving continuation pending.
     * Setup and confrontation must no longer share one generic data carrier.
     */
    sealed class SolveIntroAction : Pending() {
        abstract val stepId: String
        abstract val stage: StoryStage
        abstract val atomIndex: Int
        abstract val atomsCount: Int
        abstract val isLastHint: Boolean
        abstract val prompt: String
        abstract val gridHash12: String
        abstract val options: List<String>

        data class SetupAction(
            override val stepId: String,
            override val atomIndex: Int,
            override val atomsCount: Int,
            override val isLastHint: Boolean,
            override val prompt: String,
            override val gridHash12: String,
            override val options: List<String> = listOf(
                PendingChoiceId.SHOW_PROOF
            )
        ) : SolveIntroAction() {
            override val stage: StoryStage = StoryStage.SETUP
        }

        data class ConfrontationAction(
            override val stepId: String,
            override val atomIndex: Int,
            override val atomsCount: Int,
            override val isLastHint: Boolean,
            override val prompt: String,
            override val gridHash12: String,
            override val options: List<String> = listOf(
                PendingChoiceId.SHOW_PROOF
            )
        ) : SolveIntroAction() {
            override val stage: StoryStage = StoryStage.CONFRONTATION
        }
    }



    data class AfterResolution(
        val stepId: String,
        val prompt: String,
        val gridHash12: String,
        val options: List<String> = listOf(
            PendingChoiceId.NEXT_STEP
        )
    ) : Pending()

    /**
     * Phase 4: Commit boundary CTA (Pending-driven).
     * - The step is ready for commit; user chooses to apply now, try on paper, or ask for more explanation.
     */
    data class ApplyHintNow(
        val stepId: String,
        val prompt: String,
        val gridHash12: String,
        val options: List<String> = listOf(
            PendingChoiceId.LOCK_IT_IN
        )
    ) : Pending()




    /**
     * Phase 4 — user-owned pending projection.
     * This is NOT a controller. It is a compatibility bridge so Tick2 / CTA / pending-context
     * can reflect the active user agenda head while the true owner remains the user queue.
     */
    data class UserAgendaBridge(
        val userAgendaId: String,
        val agendaKind: String,
        val prompt: String,
        val expectedAnswerKind: String = "FREEFORM_STRUCTURED",
        val rowHint: Int? = null,
        val colHint: Int? = null,
        val digitHint: Int? = null,
        val cellRef: String? = null,
        val agendaStatus: String? = null,
        val allowsFollowUp: Boolean = false,
        val options: List<String> = emptyList()
    ) : Pending()


    /**
     * Phase 5: Free-talk detour while SOLVING.
     * We temporarily replace the solving CTA with a single “return” CTA,
     * but we keep the exact solving pending to resume deterministically.
     */
    data class ReturnToRoute(
        val resumePending: Pending?,            // usually SolveIntroAction / ApplyHintNow / AfterResolution
        val prompt: String,
        val options: List<String> = listOf(PendingChoiceId.RETURN_TO_ROUTE)
    ) : Pending()

}

data class GridSnapshot(
    val llm: LLMGridState,
    val epochMs: Long = System.currentTimeMillis()
)

data class PendingTick2(
    val toolResults: List<String>,
    val toolResultIds: List<String>,
    val listenAfter: Boolean,
    val fallbackText: String
)

data class PendingReplyV1(
    val ctx: PolicyCallCtx,
    val turnId: Long,
    val userText: String,
    val sBefore: SudoState,
    val decision: DecisionOutcomeV1,
    val factBundles: List<FactBundleV1>,

    /**
     * ✅ Phase-4 latch for deterministic ops:
     * Wait until ALL planned op toolCallIds have been observed via ToolExecuted
     * before issuing Tick2 (so bulk confirm is reliable).
     */
    val pendingToolCallIds: Set<String> = emptySet(),
    val doneToolCallIds: Set<String> = emptySet()
) {
    fun isWaiting(): Boolean = pendingToolCallIds.isNotEmpty()
    fun isDone(): Boolean = pendingToolCallIds.isNotEmpty() && doneToolCallIds.size >= pendingToolCallIds.size
    fun withDone(toolCallId: String): PendingReplyV1 =
        copy(doneToolCallIds = doneToolCallIds + toolCallId)
}

data class PendingGridFactsSnapshot(
    val ctx: PolicyCallCtx,
    val stage: String
)

data class LastCellConfirmation(
    val cellIndex: Int,              // 0..80
    val digit: Int,                  // 0..9 (0=blank)
    val changed: Boolean,
    val source: String,
    val seq: Long,
    val epochMs: Long = System.currentTimeMillis()
)

// ------------------------------------------------------------
// Phase 4: Execution evidence latch (app-owned truth)
// ------------------------------------------------------------

sealed class ExecutedActionV1 {

    /**
     * Cell edit result (includes no-op edits deterministically).
     * - from/to: digits 0..9 (0 = blank)
     */
    data class EditCell(
        val cellIndex: Int,
        val from: Int,
        val to: Int,
        val changed: Boolean
    ) : ExecutedActionV1()

    data class ClearCell(
        val cellIndex: Int,
        val from: Int
    ) : ExecutedActionV1()

    /**
     * Confirmation is NOT necessarily an edit.
     * - digit: confirmed digit 0..9 (0 = blank)
     * - changed: whether the confirmation also implied a change vs displayed state
     */
    data class ConfirmCell(
        val cellIndex: Int,
        val digit: Int,
        val changed: Boolean
    ) : ExecutedActionV1()

    /**
     * Candidates mask transition (0..0x1FF).
     */
    data class SetCandidates(
        val cellIndex: Int,
        val fromMask: Int,
        val toMask: Int
    ) : ExecutedActionV1()

    data class AddCandidate(
        val cellIndex: Int,
        val digit: Int
    ) : ExecutedActionV1()

    data class RemoveCandidate(
        val cellIndex: Int,
        val digit: Int
    ) : ExecutedActionV1()

    /**
     * Undo/Redo are acknowledged with applied flag.
     * If undo infra not present yet, applied=false is still audit-friendly.
     */
    data class Undo(
        val applied: Boolean
    ) : ExecutedActionV1()

    data class Redo(
        val applied: Boolean
    ) : ExecutedActionV1()
}

// ------------------------------------------------------------
// Phase 3: Two-queue agenda model (User agenda vs App agenda)
// ------------------------------------------------------------

// ------------------------------------------------------------
// Phase 1 — Expanded user agenda taxonomy
// Passenger-owned detours become first-class state items.
// Ownership/arbitration is NOT enforced here yet; this phase is model-only.
// ------------------------------------------------------------
sealed class UserAgendaItem {

    abstract val id: String
    abstract val createdTurnSeq: Long
    abstract val checkpointRouteId: String?
    abstract val checkpointPhase: GridPhase?
    abstract val checkpointAppAgendaKind: String?
    abstract val checkpointStepId: String?
    abstract val checkpointStoryStage: StoryStage?
    abstract val status: UserAgendaStatusV1

    // Phase 9 — mini-agenda support
    abstract val parentAgendaId: String?
    abstract val allowsFollowUp: Boolean
    abstract val lastTouchedTurnSeq: Long?

    data class Clarification(
        override val id: String,                 // stable id ("ua:...")
        val intentId: String,                    // Tick1 intent id ("t1_i0")
        val missing: List<String>,               // e.g. ["cell","digit"]
        val askedTurnSeq: Long? = null,          // when we asked the user (for expiry)
        override val createdTurnSeq: Long,
        val prompt: String,                      // what we'll ask in Tick2

        // Carry-forward semantic state for clarification chains.
        val detourQuestionClass: DetourQuestionClassV1? = null,
        val carriedCellRef: String? = null,
        val carriedDigit: Int? = null,
        val carriedRegionRef: String? = null,
        val carriedTechniqueLabel: String? = null,

        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = true,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class ProofChallenge(
        override val id: String,
        val intentId: String,
        val cellRef: String? = null,             // e.g. "r7c3"
        val digit: Int? = null,                  // e.g. 8
        val houseRef: String? = null,            // e.g. "column 3"
        val askKind: ProofChallengeAskKindV1 = ProofChallengeAskKindV1.PROVE_ELIMINATION,
        val laneHint: DetourProofChallengeLaneV1? = null,
        val rivalCellRef: String? = null,
        val rivalDigit: Int? = null,
        val claimedTechniqueId: String? = null,
        val proofObjectHint: String? = null,
        val prompt: String? = null,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class CandidateStateQuery(
        override val id: String,
        val intentId: String,
        val cellRefs: List<String> = emptyList(),
        val houseRef: String? = null,            // e.g. "row 7" | "box 7"
        val digit: Int? = null,
        val askKind: CandidateAskKindV1 = CandidateAskKindV1.CELL_CANDIDATES,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class TargetCellQuery(
        override val id: String,
        val intentId: String,
        val targetCellRef: String? = null,
        val digit: Int? = null,
        val askKind: TargetCellAskKindV1 = TargetCellAskKindV1.WHY_THIS_TARGET,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class NeighborCellQuery(
        override val id: String,
        val intentId: String,
        val anchorCellRef: String? = null,
        val neighborCellRef: String? = null,
        val askKind: NeighborAskKindV1 = NeighborAskKindV1.NEARBY_EFFECT,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class UserReasoningCheck(
        override val id: String,
        val intentId: String,
        val cellRef: String? = null,
        val proposedDigits: List<Int> = emptyList(),
        val prompt: String? = null,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class AlternativeTechniqueQuery(
        override val id: String,
        val intentId: String,
        val cellRef: String? = null,
        val regionRef: String? = null,
        val requestedTechniqueIdOrLabel: String? = null,
        val askKind: AlternativeTechniqueAskKindV1 = AlternativeTechniqueAskKindV1.IS_THIS_ALTERNATIVE_VALID,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class RouteComparisonQuery(
        override val id: String,
        val intentId: String,
        val compareScope: String? = null,        // "cell" | "box" | "row" | "local"
        val userIdeaLabel: String? = null,
        val askKind: RouteCompareAskKindV1 = RouteCompareAskKindV1.SAME_ROUTE_OR_DIFFERENT,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class RouteControl(
        override val id: String,
        val intentId: String,
        val control: UserRouteControlKindV1,
        val prompt: String? = null,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class OverlayControl(
        override val id: String,
        val intentId: String,
        val control: OverlayControlKindV1,
        val focusCellRef: String? = null,
        val regionRef: String? = null,
        val digit: Int? = null,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()

    data class GeneralQuestion(
        override val id: String,
        val intentId: String,
        val topic: String? = null,
        val prompt: String? = null,

        override val createdTurnSeq: Long,
        override val checkpointRouteId: String? = null,
        override val checkpointPhase: GridPhase? = null,
        override val checkpointAppAgendaKind: String? = null,
        override val checkpointStepId: String? = null,
        override val checkpointStoryStage: StoryStage? = null,
        override val status: UserAgendaStatusV1 = UserAgendaStatusV1.OPEN,
        override val parentAgendaId: String? = null,
        override val allowsFollowUp: Boolean = false,
        override val lastTouchedTurnSeq: Long? = null
    ) : UserAgendaItem()
}

enum class UserAgendaStatusV1 {
    OPEN,
    PARTIALLY_ANSWERED,
    RESOLVED,
    CANCELLED
}

enum class CandidateAskKindV1 {
    CELL_CANDIDATES,
    CELL_CANDIDATE_COUNT,
    HOUSE_CANDIDATE_MAP,
    DIGIT_LOCATIONS_IN_HOUSE,
    CANDIDATE_CHANGES
}

enum class DetourProofChallengeLaneV1 {
    CANDIDATE_IMPOSSIBILITY,
    CANDIDATE_POSSIBILITY,
    FORCEDNESS_OR_UNIQUENESS,
    RIVAL_COMPARISON,
    HOUSE_BLOCKER,
    TECHNIQUE_LEGITIMACY,
    ELIMINATION_LEGITIMACY,
    NON_PROOF_OR_NOT_ESTABLISHED
}

enum class ProofChallengeAskKindV1 {
    WHY_NOT_DIGIT,
    WHY_DIGIT,
    WHERE_DIGIT_CAN_STILL_GO_IN_HOUSE,
    WHY_ONLY_PLACE,
    WHY_FORCED_IN_CELL,
    WHY_THIS_TARGET,
    WHY_THIS_NOT_THAT,
    WHAT_BLOCKS_DIGIT_IN_HOUSE,
    WHY_TECHNIQUE_APPLIES,
    WHY_CURRENT_MOVE_BEFORE_OTHER_MOVE,
    PROVE_ELIMINATION,
    IS_THIS_LOCALLY_PROVED
}

enum class TargetCellAskKindV1 {
    WHY_THIS_TARGET,
    WHAT_CHANGED_IN_TARGET,
    WHAT_REMAINS_IN_TARGET,
    IS_ONLY_PLACE_FOR_DIGIT,
    IS_TARGET_DIRECT_OR_INDIRECT,
    FINAL_SURVIVING_CANDIDATE
}

enum class NeighborAskKindV1 {
    NEARBY_EFFECT,
    NEXT_BEST_NEARBY_CELL,
    STAY_IN_REGION,
    LOCAL_MOVE_SEARCH,
    COMPARE_CANDIDATES
}

enum class AlternativeTechniqueAskKindV1 {
    IS_THIS_ALTERNATIVE_VALID,
    IS_THERE_ANOTHER_TECHNIQUE_HERE,
    IS_THERE_A_SIMPLER_MOVE,
    LOCAL_ALTERNATIVE_SEARCH
}

enum class RouteCompareAskKindV1 {
    SAME_ROUTE_OR_DIFFERENT,
    SAME_TARGET_OR_DIFFERENT,
    STRONGER_OR_WEAKER,
    SHOULD_WE_SWITCH
}

enum class UserRouteControlKindV1 {
    RESUME_ROUTE,
    RETURN_TO_ORIGINAL_MOVE,
    CONTINUE_ROUTE,
    PAUSE_ROUTE,
    ABANDON_ROUTE
}

enum class OverlayControlKindV1 {
    PRESERVE_CURRENT,
    STAY_ON_CURRENT_REGION,
    HIGHLIGHT_CELL,
    HIGHLIGHT_BLOCKER,
    HIGHLIGHT_ALTERNATIVE_AREA,
    COMPARE_MY_AREA_TO_YOURS
}

sealed class AppAgendaItem {

    // ------------------------------------------------------------
    // CONFIRMING phase (TO-BE agenda items)
    // ------------------------------------------------------------

    /**
     * AGENDA 1: RETAKE decision path (strong/soft/none).
     * User chooses: retake OR stay and fix issues one-by-one.
     */
    data class RetakeDecision(
        val strength: String,          // "soft"|"strong"
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    /**
     * AGENDA 2: MISMATCH review (one at a time).
     * Propose an edit for a wrong filled-in answer vs deduced expected value.
     */
    data class MismatchReview(
        val cellIndex: Int,
        val scannedDigit: Int,
        val expectedDigit: Int,
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    /**
     * AGENDA 3: CONFLICT review (one at a time).
     * Ask user to confirm what is on paper/book for this cell (and optionally provide corrected value).
     */
    data class ConflictReview(
        val cellIndex: Int,
        val reason: String,            // "conflict" | "low_confidence" | "unresolved"
        val createdTurnSeq: Long
    ) : AppAgendaItem()



    /**
     * AGENDA 4: exact-match signoff gate.
     * Triggered when the on-screen grid is now uniquely solvable, but we still require
     * the human to confirm that paper/book and screen match 100% before solving begins.
     */
    data class ConfirmValidationExactMatch(
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    /**
     * AGENDA 5: VISUAL VERIFICATION from user (100% match check).
     * Triggered when mismatch/conflict lists are empty but solvability is still not "unique".
     */
    data class VisualVerificationRequest(
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    /**
     * AGENDA 6: Recommend capturing a different grid (not uniquely solvable after full verification).
     */
    data class RecommendDifferentGrid(
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    /**
     * AGENDA 7: Finalize grid validation presentation, then ask to start solving.
     * This may only occur after the user has already confirmed that paper/book and
     * screen match exactly.
     */
    data class FinalizeValidationStartSolving(
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()



    // ------------------------------------------------------------
    // Phase 0 — SOLVING agenda taxonomy (scaffold; wired in Phase 1+)
    // ------------------------------------------------------------
    sealed class Solving : AppAgendaItem() {
        abstract val agendaId: String
        abstract val createdAtMs: Long
        abstract val createdTurnSeq: Long
        abstract val stepIndex: Int
        abstract val route: ActiveRouteContextV2

        data class StepIntro(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val techniqueId: String? = null,
            val techniqueName: String? = null,
            val mission: String? = null,
            val allowAskTechnique: Boolean = true,
            val allowDetour: Boolean = true,
            val allowWalkback: Boolean = true
        ) : Solving()

        /**
         * Confrontation-only walk carrier.
         * Setup must remain on StepIntro; resolution must move to commit/post-commit carriers.
         */
        data class AtomWalk(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val atomIndex: Int,
            val atomCount: Int,
            val pacing: AtomPacing = AtomPacing.ONE_ATOM_PER_TURN,
            val allowDetour: Boolean = true,
            val allowWalkback: Boolean = true
        ) : Solving() {
            enum class AtomPacing { ONE_ATOM_PER_TURN, TWO_ATOMS_PER_TURN, UNTIL_COMMIT }
        }

        data class CommitDecision(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val commitAtomIndex: Int,
            val opsCount: Int,
            val allowTrySelf: Boolean = true,
            val allowExplainMore: Boolean = true
        ) : Solving()

        data class CommitApply(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val commitReason: String
        ) : Solving()

        data class PostCommitChoice(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val allowTeachTechnique: Boolean = true,
            val allowDetour: Boolean = true
        ) : Solving()

        data class TechniqueTeach(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val techniqueId: String,
            val techniqueName: String? = null,
            val family: String? = null,
            val definitionWhat: String? = null,
            val definitionWhy: String? = null,
            val definitionWhen: String? = null,
            val definitionHow: String? = null
        ) : Solving()

        data class DetourFreeTalk(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val detourTopicHint: String? = null
        ) : Solving()

        data class Walkback(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val undoOpsCount: Int
        ) : Solving()

        data class RecoveryRecomputeStep(
            override val agendaId: String,
            override val createdAtMs: Long,
            override val createdTurnSeq: Long,
            override val stepIndex: Int,
            override val route: ActiveRouteContextV2,
            val reason: RecoveryReason,
            val allowRetake: Boolean = true,
            val allowRecompute: Boolean = true
        ) : Solving() {
            enum class RecoveryReason { STEP_STALE, STEP_UNAVAILABLE, ROUTE_HASH_MISMATCH }
        }
    }

    // ------------------------------------------------------------
    // SOLVING phase placeholder (legacy)
    // ------------------------------------------------------------
    // Removed: Phase 3+ uses AppAgendaItem.Solving.* exclusively.

    // ------------------------------------------------------------
    // Legacy items (kept to avoid breaking older codepaths; will be removed after migration)
    // ------------------------------------------------------------
    @Deprecated("Use RetakeDecision instead")
    data class ProposeRetake(
        val strength: String,
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    @Deprecated("Use MismatchReview instead")
    data class ProposeEdit(
        val cellIndex: Int,
        val scannedDigit: Int,
        val expectedDigit: Int,
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()

    @Deprecated("Use ConflictReview instead")
    data class AskConfirmCell(
        val cellIndex: Int,
        val reason: String,
        val createdTurnSeq: Long
    ) : AppAgendaItem()
}

data class UserPrefsV1(
    val language: String = "auto",             // "auto"|"en"|"fr"
    val notation: String = "rXcY",             // "rXcY"|"A1"
    val evidenceVerbosity: String = "normal",  // "light"|"normal"|"deep"
    val hintLevel: String = "gentle",          // "minimal"|"gentle"|"explicit"
    val teachMode: Boolean = false,
    val fastMode: Boolean = false,
    val oneQuestionMax: Boolean = true,
    val onlyValidate: Boolean = false,
    val onlySolve: Boolean = false
)

// ------------------------------------------------------------
// Phase 6: Route context (detours + return-to-route without recompute)
// ------------------------------------------------------------
data class ActiveRouteContext(
    val stepId: String,
    val gridHash12: String,
    val journeyStepNumber: Int,

    // Phase 8: atom-native cursor
    val atomIndex: Int,
    val atomsCount: Int,

    val techniqueId: String? = null,
    val targetCellIndex: Int? = null
)

enum class SolvingRailOwnerV1 {
    APP_MAINLINE,
    USER_DETOUR,
    NONE
}

enum class SolvingRailBoundaryV1 {
    NONE,
    SETUP,
    CONFRONTATION,
    COMMIT_DECISION,
    POST_COMMIT,
    NEXT_STEP_IN_FLIGHT
}

data class SolvingRailStateV1(
    val stepId: String? = null,
    val gridHash12: String? = null,
    val stage: StoryStage? = null,
    val boundary: SolvingRailBoundaryV1 = SolvingRailBoundaryV1.NONE,
    val owner: SolvingRailOwnerV1 = SolvingRailOwnerV1.NONE,
    val journeyStepNumber: Int? = null,
    val hasNextStepInFlight: Boolean = false
)

enum class CanonicalSolvingPositionKindV1 {
    SETUP,
    CONFRONTATION,
    RESOLUTION_COMMIT,
    RESOLUTION_POST_COMMIT
}

enum class CanonicalSolvingHeadKindV1 {
    NONE,
    STEP_INTRO,
    ATOM_WALK,
    COMMIT_DECISION,
    COMMIT_APPLY,
    POST_COMMIT_CHOICE
}

enum class CanonicalSolvingPendingKindV1 {
    NONE,
    SOLVE_INTRO_ACTION,
    APPLY_HINT_NOW,
    AFTER_RESOLUTION,
    SOLVE_STEP_MISSING,
    OTHER
}

data class CanonicalSolvingStepPositionV1(
    val kind: CanonicalSolvingPositionKindV1,
    val stepId: String? = null,
    val gridHash12: String? = null,
    val stage: StoryStage,
    val headKind: CanonicalSolvingHeadKindV1 = CanonicalSolvingHeadKindV1.NONE,
    val pendingKind: CanonicalSolvingPendingKindV1 = CanonicalSolvingPendingKindV1.NONE,
    val focusAtomIndex: Int? = null,
    val atomsCount: Int? = null,
    val readyForCommit: Boolean = false
)

enum class BoundaryGuardModeV1 {
    NORMAL_PROGRESS,
    RESTORE_PAUSED_ROUTE,
    RECOMPUTE_FROM_CURRENT_TRUTH
}

enum class BoundaryResumePolicyV1 {
    RESTORE_PAUSED_ROUTE,
    RECOMPUTE_FROM_CURRENT_TRUTH
}

enum class ResolutionCommitOutcomeV1 {
    NONE,
    APP_AUTO_COMMIT,
    MANUAL_APPLY_REQUIRED,
    COMMIT_UNAVAILABLE
}

enum class EffectiveBoundaryBeforeSourceV1 {
    RAW_BEFORE,
    CHECKPOINT_BEFORE,
    RESTORED_ROUTE_BEFORE,
    NONE_RECOMPUTE
}

data class BoundaryGuardContextV1(
    val mode: BoundaryGuardModeV1,
    val rawBeforePos: CanonicalSolvingStepPositionV1? = null,
    val checkpointBeforePos: CanonicalSolvingStepPositionV1? = null,
    val restoredRouteBeforePos: CanonicalSolvingStepPositionV1? = null,
    val effectiveBeforePos: CanonicalSolvingStepPositionV1? = null,
    val afterPos: CanonicalSolvingStepPositionV1? = null,
    val beforeSource: EffectiveBoundaryBeforeSourceV1 = EffectiveBoundaryBeforeSourceV1.RAW_BEFORE,
    val sameStepGuardEnabled: Boolean = true,
    val trustSource: String = "raw_before_default"
)

data class SudoState(
    val sessionId: String,

    // Tick2 consistency
    val systemPrompt: String = "",

    val mode: SudoMode = SudoMode.FREE_TALK,

    val grid: GridSnapshot? = null,
    val pending: Pending? = null,

    // AGENDA 4 / 5 support: user explicitly confirmed 100% visual match.
    val userConfirmedPerfectMatch: Boolean = false,

    // AGENDA 1 support: once the user continues (doesn't choose retake), never re-ask retake for this grid.
    val retakeDismissed: Boolean = false,

    // Phase 2 DecisionMemory: explicit retake fork decision for this grid (monotonic waypoint commit).
    val retakeDecision: RetakeDecisionV1 = RetakeDecisionV1.UNDECIDED,

    // Phase 3: queues
    val userAgendaQueue: List<UserAgendaItem> = emptyList(),

    val appAgendaQueue: List<AppAgendaItem> = emptyList(),

    val phase: GridPhase = GridPhase.CONFIRMING,

    val lastUserText: String? = null,
    val lastAssistantText: String? = null,

    // ✅ Phase-2: deterministic binder input (the exact TurnContextV1 JSON sent at Tick-1)
    val lastTick1TurnCtxJson: String? = null,

    val lastConfirmation: LastCellConfirmation? = null,

    val focusCellIndex: Int? = null,

    val repairAttempt: Int = 0,

    val turnSeq: Long = 0L,
    val policyReqSeq: Long = 0L,

    val pendingTick2: PendingTick2? = null,

    // ✅ v1 (Meaning→Decision→Reply) latch
    val pendingReplyV1: PendingReplyV1? = null,

    val pendingGridFactsSnapshot: PendingGridFactsSnapshot? = null,

    val pendingPhaseAfterSeal: GridPhase? = null,
    val pendingSealToolResultId: String? = null,

    // ----------------------------
    // Phase 4: executed action evidence (authoritative for narration + audit)
    // ----------------------------
    val lastExecutedActions: List<ExecutedActionV1> = emptyList(),

    /**
     * Optional convenience: bulk summary count (can be derived from lastExecutedActions.size).
     * Keep in state so Tick2 can narrate “confirmed 9 cells” without re-deriving.
     */
    val lastBulkCount: Int = 0,

    // ----------------------------
    // SOLVING: engine-derived next step (authoritative)
    // ----------------------------
    val solveStepJson: String? = null,
    val solveStepGridHash12: String? = null,

    /**
     * Short human-readable summary of the last engine step (1–3 lines),
     * derived deterministically from solveStepJson / coachPlan.
     * This is what we inject into GRID_CONTEXT in SOLVING.
     */
    val solveStepSummary: String? = null,

    /**
     * Stable id for the step, referenced by CTA tools.
     * Derived in-conductor on SolveStepUpdated.
     */
    val solveStepId: String? = null,

    /**
     * Monotonic counter to disambiguate repeated solve steps for same grid hash.
     */
    val solveStepSeq: Long = 0L,

    /**
     * ✅ Phase 2: Sticky guidance mode for how to consume the CURRENT step.
     * UNSET resets on new SolveStepUpdated (new step).
     */
    val solveGuidanceMode: SolveGuidanceModeV1 = SolveGuidanceModeV1.UNSET,

    // ----------------------------
    // Phase 4: skip-ahead state (deterministic)
    // ----------------------------
    /**
     * When > 0, we are auto-applying N revealed steps (no Tick2 speech) and then
     * will present the next computed step normally.
     */
    val solveSkipRemaining: Int = 0,

    /**
     * Total requested skip count (for narration/telemetry if needed).
     */
    val solveSkipTotal: Int = 0,

    /**
     * Phase 4 second half: skip narration range.
     * Example: user asks "skip next 3" at journey step 44 => start=44, end=47.
     */
    val solveSkipStartJourneyStep: Int = 0,
    val solveSkipEndJourneyStep: Int = 0,

    // ----------------------------
    // SOLVING: CoachPlan (typed, authoritative)
    // ----------------------------

    /**
     * ✅ Authoritative typed plan derived deterministically from solveStepJson.
     * The LLM never invents hint counts or "last hint" — it only verbalizes plan content.
     */
    val coachPlan: CoachPlan? = null,

    // ------------------------------------------------------------
    // Phase 6: Route context (persisted)
    // ------------------------------------------------------------
    val activeRouteContext: ActiveRouteContext? = null,


    // ------------------------------------------------------------
    // Batch D: authoritative solving continuity rail
    // One compact summary of the current app-owned solving route state.
    // ------------------------------------------------------------
    val solvingRailStateV1: SolvingRailStateV1? = null,


    // ------------------------------------------------------------
    // Phase 2 — Persisted route checkpoint
    // Exact paused location for detours + return-to-route.
    // ------------------------------------------------------------
    val routeCheckpointV1: RouteCheckpointV1? = null,


    // ------------------------------------------------------------
    // Phase 3 — authoritative owner at the Tick1 -> Tick2 boundary
    // USER_OWNER wins whenever the user agenda is non-empty after Tick1.
    // ------------------------------------------------------------
    val activeTurnOwnerV1: TurnOwnerV1 = TurnOwnerV1.NONE,

    // ------------------------------------------------------------
    // Wave 1 — constitutional turn authority snapshot.
    // This does NOT replace activeTurnOwnerV1 yet.
    // For now it is the formal place where later waves will record:
    //   - the sovereign owner for this turn,
    //   - why it won,
    //   - what evidence competed,
    //   - whether pending was suspended,
    //   - whether a route jump / repair / detour was requested.
    // ------------------------------------------------------------
    val activeTurnAuthorityDecisionV1: TurnAuthorityDecisionV1 =
        TurnAuthorityDecisionV1(
            owner = TurnAuthorityOwnerV1.NONE,
            reason = "state_default"
        ),

    // ------------------------------------------------------------
    // Wave 7 — explicit route boundary snapshot.
    // Tracks whether the current checkpoint is paused, blocked, re-entered,
    // or legally released to the next step.
    // ------------------------------------------------------------
    val activeRouteBoundaryDecisionV1: RouteBoundaryDecisionV1 =
        RouteBoundaryDecisionV1(
            status = RouteBoundaryStatusV1.NONE,
            reason = "state_default"
        ),

    // ------------------------------------------------------------
    // Phase 6 — authoritative overlay ownership
    // Overlay follows the active owner, not a loose detour override.
    // ------------------------------------------------------------
    val activeOverlayOwnerV1: OverlayOwnerV1 = OverlayOwnerV1.NONE,


    val activeAppliedOverlayFrameIdsV1: List<String> = emptyList(),
    val activeOverlayReasonV1: String? = null,


    // ------------------------------------------------------------
    // Phase 0 — Detour state scaffold (persisted; behavior added later)
    // ------------------------------------------------------------
    val detourState: DetourStateV1? = null,

    val detourTurnHistoryV1: List<DetourTurnRecordV1> = emptyList(),

    // ------------------------------------------------------------
    // Phase 1: Narrative Story context (persisted)
    // ------------------------------------------------------------
    val narrativeStory: NarrativeStoryContext? = null,

    // ------------------------------------------------------------
    // Wave 1 permanent-design: native detour narrative context.
    // This is the detour sibling of NarrativeStoryContext.
    // It is introduced as native state now and wired in later phases.
    // ------------------------------------------------------------
    val detourNarrativeContextV1: DetourNarrativeContextV1? = null,

    // ------------------------------------------------------------
    // North Star: persisted canonical solving boundary.
    // This is the app-owned authoritative position for the current solving step.
    // Legacy / recovery code may still recompute from carriers when absent or stale.
    // ------------------------------------------------------------
    val canonicalSolvingPositionV1: CanonicalSolvingStepPositionV1? = null,

    /**
     * ✅ Phase 3: deterministic anti-repeat guard.
     * Stores the signature of the last Tick2 story emission:
     * (stepId + stage + atoms_in_scope + required_end_cta_options).
     * If the next Tick2 would emit the same signature (and user didn’t ask to repeat),
     * the conductor forces story progression.
     */
    val lastStorySignatureV1: String? = null,

    // ------------------------------------------------------------
    // ✅ Phase 5 — Step story scorecard (North Star conformance)
    // Tracks what was actually DELIVERED to the user for the current stepId.
    // ------------------------------------------------------------
    val storyScoreStepIdV1: String? = null,          // step_id being scored
    val storyDeliveredSetupV1: Boolean = false,      // delivered a SETUP/spotlight turn
    val storyDeliveredProofMaxAtomV1: Int = 0,       // highest proof atom delivered (>=1)
    val storyDeliveredCommitV1: Boolean = false,     // delivered a RESOLUTION/commit turn

    /**
     * Evidence/binding for the current coachPlan (debug + stale-drop parity).
     */
    val coachPlanGridHash12: String? = null,
    val coachPlanStepId: String? = null,
    val coachPlanSha12: String? = null,

    /**
     * Legacy/telemetry/debug: keep JSON if you want to log it or render in dev tools.
     * Not authoritative once coachPlan is present.
     */
    val coachPlanJson: String? = null,

    /**
     * Phase 8: Atom-native proof cursor.
     *
     * - We no longer drive the conversation by "atomIndex/atomsCount".
     * - Proof beats are narrative atoms 1..(atomsCount-2).
     * - Commit is atom (atomsCount-1).
     */
    val solveAtomIndex: Int = 0,      // 0=setup spotlight; 1..(N-2)=proof beats; (N-1)=commit
    val solveAtomsCount: Int = 0,     // copied from narrative_atoms_v1.atoms.length

    val solveStepStatus: SolveStepStatus = SolveStepStatus.IDLE,
    val pendingSolveStepAutocoach: PendingSolveStepAutocoach? = null,

    val solveOverlayVisible: Boolean = false,
    val solveOverlayStyle: String = "full",

    // ----------------------------
    // DecisionTrace (app-only, debug/audit)
    // ----------------------------
    val lastMeaningIntentType: IntentTypeV1? = null,
    val lastMeaningConfidence: Double? = null,
    val lastMeaningNeedsClarification: Boolean? = null,

    val lastDecisionKind: DecisionKindV1? = null,
    val lastFactBundleTypes: List<String> = emptyList(),

    val lastReplyLen: Int = 0,

    // Optional timing breadcrumbs (epoch ms; set by Store/Runner)
    val lastTick1StartMs: Long? = null,
    val lastTick1EndMs: Long? = null,
    val lastDecisionStartMs: Long? = null,
    val lastDecisionEndMs: Long? = null,
    val lastTick2StartMs: Long? = null,
    val lastTick2EndMs: Long? = null,

    // Milestone 6: SOLVING fact bundles computed when a step becomes READY.
    // Tick2 should consume these directly (no extra engine calls).
    val solvingReadyFactBundlesV1: List<FactBundleV1> = emptyList(),

// Phase 0: last emitted TurnContractSnapshot (debug/audit; used for invariants)
    val lastTurnContractSnapshot: TurnContractSnapshotV1? = null,

    val prefs: UserPrefsV1 = UserPrefsV1()
)


fun SudoState.isCommitBoundaryPendingV1(): Boolean =
    pending is Pending.ApplyHintNow

fun SudoState.isPostCommitPendingV1(): Boolean =
    pending is Pending.AfterResolution

fun SudoState.hasExecutedRevealEditV1(): Boolean {
    val rev = coachPlan?.reveal ?: return false
    val idx = rev.placementCellIndex
    val digit = rev.placementDigit
    if (idx !in 0..80 || digit !in 1..9) return false

    return lastExecutedActions.any { act ->
        act is ExecutedActionV1.EditCell &&
                act.cellIndex == idx &&
                act.to == digit
    }
}