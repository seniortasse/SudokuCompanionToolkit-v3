package com.contextionary.sudoku.conductor.policy

import com.contextionary.sudoku.conductor.ClarifyKind
import com.contextionary.sudoku.conductor.policy.DetourQuestionClassV1
import com.contextionary.sudoku.conductor.GridPhase
import com.contextionary.sudoku.conductor.policy.IntentTypeV1
import com.contextionary.sudoku.conductor.Pending
import com.contextionary.sudoku.conductor.SudoConductor
import com.contextionary.sudoku.conductor.SudoState
import com.contextionary.sudoku.conductor.StoryStage
import com.contextionary.sudoku.conductor.UserAgendaItem

/**
 * Wave 1 — Clarification sovereignty scaffold.
 *
 * This file is intentionally behavior-neutral in Patch Series 1.
 *
 * Goals of this first series:
 * - create a single future home for clarification policy
 * - define stable data models for clarification admission / carry / prompt / pending
 * - avoid any live routing changes until later patch series migrate logic here
 *
 * Non-goals in this series:
 * - no change to current conductor behavior
 * - no change to reply demand resolution
 * - no change to prompt composition
 */
object ClarificationKernelV1 {

    /**
     * Patch Series 1:
     * This is a placeholder entry point only.
     *
     * Later patch series will replace conductor-local clarification admission with
     * calls into this method, but for now we keep it inert to avoid behavior drift.
     */
    fun assess(
        s: SudoState,
        intents: List<IntentV1>,
        phase: GridPhase,
        userText: String?
    ): ClarificationAssessmentV1 {
        return ClarificationAssessmentV1.NotNeeded(
            reason = "wave1_scaffold_behavior_neutral"
        )
    }

    /**
     * Patch Series 1:
     * Placeholder for the future "does this new user utterance satisfy the active
     * clarification?" check.
     */
    fun assessResolution(
        pending: Pending.AskClarification?,
        intents: List<IntentV1>,
        userText: String?
    ): ClarificationResolutionAssessmentV1 {
        if (pending == null) {
            return ClarificationResolutionAssessmentV1.NotResolved(
                reason = "no_pending_ask_clarification"
            )
        }

        val syntheticHead =
            UserAgendaItem.Clarification(
                id = "pending:ask_clarification",
                intentId = "pending:ask_clarification",
                missing = buildList {
                    when (pending.kind) {
                        ClarifyKind.ROW -> add("row")
                        ClarifyKind.COL -> add("column")

                        ClarifyKind.POSITION -> {
                            if (pending.rowHint == null) add("row")
                            if (pending.colHint == null) add("column")
                        }

                        ClarifyKind.DIGIT -> add("digit")
                        ClarifyKind.YESNO -> add("yes_no")
                        ClarifyKind.WORKFLOW -> add("workflow")
                    }
                },
                askedTurnSeq = null,
                createdTurnSeq = 0L,
                prompt = pending.prompt
            )

        return assessResolutionForHead(
            head = syntheticHead,
            intents = intents
        )
    }

    fun assessResolutionForHead(
        head: UserAgendaItem.Clarification,
        intents: List<IntentV1>
    ): ClarificationResolutionAssessmentV1 {
        val explicit =
            intents.any { it.addressesUserAgendaId?.trim()?.takeIf { v -> v.isNotBlank() } == head.id }
        if (explicit) {
            return ClarificationResolutionAssessmentV1.Resolved(
                recoveredSlots = mapOf(
                    ClarificationSlotV1.REFERENCE_TARGET to head.id
                ),
                reason = "explicit_user_agenda_address"
            )
        }

        val needsCell =
            head.missing.any {
                it.equals("cell", true) ||
                        it.equals("row", true) ||
                        it.equals("col", true) ||
                        it.equals("column", true) ||
                        it.equals("cell_a", true) ||
                        it.equals("cell_b", true)
            }

        val needsDigit =
            head.missing.any {
                it.equals("digit", true) ||
                        it.equals("value", true) ||
                        it.equals("digits", true) ||
                        it.equals("regionDigits", true)
            }

        val needsYesNo =
            head.missing.any {
                it.equals("yes_no", true) ||
                        it.equals("yesno", true)
            }

        val hasAnyCell = intents.any { it.cellIndexOrNullCompatV1() != null }
        val hasAnyDigit = intents.any { it.digitOrNullCompatV1() != null || !it.payload.digits.isNullOrEmpty() }
        val hasYesNo =
            intents.any {
                it.type == IntentTypeV1.CONFIRM_YES ||
                        it.type == IntentTypeV1.CONFIRM_NO
            }
        val hasNonFreeTalk = intents.any { it.type != IntentTypeV1.FREE_TALK }

        val recovered = linkedMapOf<ClarificationSlotV1, String>()
        if (hasAnyCell) recovered[ClarificationSlotV1.CELL] = "present"
        if (hasAnyDigit) recovered[ClarificationSlotV1.DIGIT] = "present"
        if (hasYesNo) recovered[ClarificationSlotV1.YES_NO] = "present"

        val cellOk = !needsCell || hasAnyCell
        val digitOk = !needsDigit || hasAnyDigit
        val yesNoOk = !needsYesNo || hasYesNo

        return when {
            hasNonFreeTalk && cellOk && digitOk && yesNoOk ->
                ClarificationResolutionAssessmentV1.Resolved(
                    recoveredSlots = recovered,
                    reason = "heuristic_missing_slots_satisfied"
                )

            recovered.isNotEmpty() ->
                ClarificationResolutionAssessmentV1.PartiallyResolved(
                    recoveredSlots = recovered,
                    reason = "heuristic_missing_slots_partial"
                )

            else ->
                ClarificationResolutionAssessmentV1.NotResolved(
                    reason = "heuristic_missing_slots_unsatisfied"
                )
        }
    }

    fun clarificationPendingAfterConsumption(
        pending: Pending?
    ): Pending? =
        when (pending) {
            is Pending.AskClarification -> null
            else -> pending
        }

    fun isClarificationSovereign(
        pendingAfter: Pending?,
        userAgendaHead: UserAgendaItem?,
        decisionAskedClarification: Boolean,
        decisionNeedsClarification: Boolean
    ): Boolean {
        val materializedPending =
            (pendingAfter as? Pending.AskClarification)?.prompt?.isNotBlank() == true

        return materializedPending ||
                decisionAskedClarification ||
                decisionNeedsClarification ||
                userAgendaHead is UserAgendaItem.Clarification
    }

    fun clarificationSovereigntyReason(
        pendingAfter: Pending?,
        userAgendaHead: UserAgendaItem?,
        decisionAskedClarification: Boolean,
        decisionNeedsClarification: Boolean
    ): String =
        when {
            (pendingAfter as? Pending.AskClarification)?.prompt?.isNotBlank() == true ->
                "clarification_sovereignty_from_materialized_pending"

            decisionAskedClarification ->
                "clarification_sovereignty_from_decision_kind"

            decisionNeedsClarification ->
                "clarification_sovereignty_from_decision_flag"

            userAgendaHead is UserAgendaItem.Clarification ->
                "clarification_sovereignty_from_user_agenda_head"

            else ->
                "clarification_sovereignty_fallback"
        }

    fun isClarificationSatisfiedByOps(
        head: UserAgendaItem.Clarification,
        ops: List<SudoConductor.PlannedOpV1>
    ): Boolean {
        if (ops.isEmpty()) return false

        val needsCell =
            head.missing.any {
                it.equals("cell", ignoreCase = true) ||
                        it.equals("row", ignoreCase = true) ||
                        it.equals("column", ignoreCase = true) ||
                        it.equals("col", ignoreCase = true) ||
                        it.equals("target", ignoreCase = true) ||
                        it.equals("scope", ignoreCase = true)
            }

        val needsDigit =
            head.missing.any { it.equals("digit", ignoreCase = true) || it.equals("digits", ignoreCase = true) }

        return ops.any { op ->
            when (op) {
                is SudoConductor.PlannedOpV1.ApplyCellEdit ->
                    (!needsCell || op.cellIndex in 0..80) && (!needsDigit || op.digit in 0..9)

                is SudoConductor.PlannedOpV1.ConfirmCellValue ->
                    (!needsCell || op.cellIndex in 0..80) && (!needsDigit || op.digit in 0..9)

                is SudoConductor.PlannedOpV1.AddCandidate ->
                    (!needsCell || op.cellIndex in 0..80) && (!needsDigit || op.digit in 1..9)

                is SudoConductor.PlannedOpV1.RemoveCandidate ->
                    (!needsCell || op.cellIndex in 0..80) && (!needsDigit || op.digit in 1..9)

                is SudoConductor.PlannedOpV1.SetCandidatesMask ->
                    (!needsCell || op.cellIndex in 0..80) && (!needsDigit || op.mask != 0)

                is SudoConductor.PlannedOpV1.FinalizeValidationPresentation ->
                    !needsCell && !needsDigit

                is SudoConductor.PlannedOpV1.Undo,
                is SudoConductor.PlannedOpV1.Redo ->
                    head.missing.isEmpty()

                is SudoConductor.PlannedOpV1.ComputeSolveStep ->
                    false
            }
        }
    }

    // ============================================================
    // Patch Series 2 — centralized hint shaping + prompt wording
    // ============================================================

    fun cellIndexHintFromPending(pending: Pending?): Int? {
        val p = pending as? Pending.AskClarification ?: return null
        val row = p.rowHint?.takeIf { it in 1..9 } ?: return null
        val col = p.colHint?.takeIf { it in 1..9 } ?: return null
        return ((row - 1) * 9) + (col - 1)
    }

    fun digitHintFromPending(pending: Pending?): Int? =
        (pending as? Pending.AskClarification)?.digitHint?.takeIf { it in 0..9 }

    fun rowColHintsFromCellIndex(cellIndex: Int?): Pair<Int?, Int?> =
        if (cellIndex == null) {
            null to null
        } else {
            ((cellIndex.coerceIn(0, 80) / 9) + 1) to ((cellIndex.coerceIn(0, 80) % 9) + 1)
        }

    fun buildRepairClarificationPrompt(
        actionVerb: String,
        rowHint: Int?,
        colHint: Int?,
        needCell: Boolean,
        needDigit: Boolean
    ): String =
        when {
            needCell && needDigit && rowHint != null && colHint != null ->
                "I caught row $rowHint, column $colHint. What digit should go there?"

            needCell && needDigit && rowHint != null && colHint == null ->
                "I caught row $rowHint. Which column and digit do you mean?"

            needCell && needDigit && rowHint == null && colHint != null ->
                "I caught column $colHint. Which row and digit do you mean?"

            needCell && needDigit ->
                "Which row, column, and digit do you mean?"

            needCell && rowHint != null && colHint == null ->
                "I caught row $rowHint. Which column do you mean?"

            needCell && rowHint == null && colHint != null ->
                "I caught column $colHint. Which row do you mean?"

            needCell ->
                "Which row and column do you mean?"

            needDigit && rowHint != null && colHint != null ->
                "What digit do you want at row $rowHint, column $colHint?"

            needDigit ->
                "Which digit do you mean?"

            else ->
                "Could you clarify that?"
        }

    fun promptForFocusCell(): String =
        "Which row and column do you mean?"

    fun promptForRegionKind(): String =
        "Do you mean a row, a column, or a box?"

    fun promptForRegionDigits(): String =
        "Please tell me the 9 digits for that region, like 951873264."

    fun promptForCandidateAdd(): String =
        "Which row and column should I add those candidates to, and which digits are they?"

    fun promptForCandidateRemove(): String =
        "Which row and column should I remove those candidates from, and which digits are they?"

    fun promptForCandidateSet(): String =
        "Which row and column do you mean, and what full candidate set should I use there?"

    fun promptForModeChange(): String =
        "Do you want fast mode or teach mode?"

    // ============================================================
    // Patch Series 3 — explicit action clarification admission
    // ============================================================

    data class ActionClarificationSpecV1(
        val missing: List<String>,
        val prompt: String,
        val kind: ClarifyKind,
        val rowHint: Int? = null,
        val colHint: Int? = null,
        val digitHint: Int? = null
    )

    data class FocusCellAdmissionV1(
        val cellIndex: Int?,
        val clarification: ActionClarificationSpecV1? = null
    )

    data class FocusRegionAdmissionV1(
        val region: RegionRefV1?,
        val clarification: ActionClarificationSpecV1? = null
    )

    data class CellOnlyAdmissionV1(
        val cellIndex: Int?,
        val rowHint: Int?,
        val colHint: Int?,
        val clarification: ActionClarificationSpecV1? = null
    )

    data class CellDigitAdmissionV1(
        val cellIndex: Int?,
        val digit: Int?,
        val rowHint: Int?,
        val colHint: Int?,
        val clarification: ActionClarificationSpecV1? = null
    )

    data class RegionAdmissionV1(
        val region: RegionRefV1?,
        val clarification: ActionClarificationSpecV1? = null
    )

    data class RegionDigitsAdmissionV1(
        val region: RegionRefV1?,
        val digits: List<Int>?,
        val clarification: ActionClarificationSpecV1? = null
    )

    data class CandidateDigitsAdmissionV1(
        val cellIndex: Int?,
        val digits: List<Int>?,
        val clarification: ActionClarificationSpecV1? = null
    )

    fun assessFocusCellAction(intent: IntentV1): FocusCellAdmissionV1 {
        val idx = intent.cellIndexOrNullCompatV1()
        return if (idx == null) {
            FocusCellAdmissionV1(
                cellIndex = null,
                clarification = ActionClarificationSpecV1(
                    missing = listOf("cell"),
                    prompt = promptForFocusCell(),
                    kind = ClarifyKind.POSITION
                )
            )
        } else {
            FocusCellAdmissionV1(cellIndex = idx)
        }
    }

    fun assessFocusRegionAction(intent: IntentV1): FocusRegionAdmissionV1 {
        val region = intent.regionOrNullCompatV1()
        return if (region == null) {
            FocusRegionAdmissionV1(
                region = null,
                clarification = ActionClarificationSpecV1(
                    missing = listOf("region"),
                    prompt = promptForRegionKind(),
                    kind = ClarifyKind.POSITION
                )
            )
        } else {
            FocusRegionAdmissionV1(region = region)
        }
    }

    fun assessEditCellAction(
        s: SudoState,
        intent: IntentV1
    ): CellDigitAdmissionV1 =
        assessCellDigitRepairAction(
            s = s,
            intent = intent,
            actionVerb = "update"
        )

    fun assessConfirmCellToDigitAction(
        s: SudoState,
        intent: IntentV1
    ): CellDigitAdmissionV1 =
        assessCellDigitRepairAction(
            s = s,
            intent = intent,
            actionVerb = "confirm"
        )

    fun assessClearCellAction(
        s: SudoState,
        intent: IntentV1
    ): CellOnlyAdmissionV1 {
        val idx = mergedRepairCellIndexCompatV1(s, intent)
        val (rowHint, colHint) = rowColHintsFromCellIndex(idx)

        return if (idx == null) {
            CellOnlyAdmissionV1(
                cellIndex = null,
                rowHint = rowHint,
                colHint = colHint,
                clarification = ActionClarificationSpecV1(
                    missing = buildMissingCellParts(rowHint, colHint),
                    prompt = buildRepairClarificationPrompt(
                        actionVerb = "clear",
                        rowHint = rowHint,
                        colHint = colHint,
                        needCell = true,
                        needDigit = false
                    ),
                    kind = ClarifyKind.POSITION,
                    rowHint = rowHint,
                    colHint = colHint
                )
            )
        } else {
            CellOnlyAdmissionV1(
                cellIndex = idx,
                rowHint = rowHint,
                colHint = colHint
            )
        }
    }

    fun assessConfirmCellAsIsAction(
        s: SudoState,
        intent: IntentV1
    ): CellOnlyAdmissionV1 {
        val idx = mergedRepairCellIndexCompatV1(s, intent)
        val (rowHint, colHint) = rowColHintsFromCellIndex(idx)

        return if (idx == null) {
            CellOnlyAdmissionV1(
                cellIndex = null,
                rowHint = rowHint,
                colHint = colHint,
                clarification = ActionClarificationSpecV1(
                    missing = buildMissingCellParts(rowHint, colHint),
                    prompt = buildRepairClarificationPrompt(
                        actionVerb = "confirm",
                        rowHint = rowHint,
                        colHint = colHint,
                        needCell = true,
                        needDigit = false
                    ),
                    kind = ClarifyKind.POSITION,
                    rowHint = rowHint,
                    colHint = colHint
                )
            )
        } else {
            CellOnlyAdmissionV1(
                cellIndex = idx,
                rowHint = rowHint,
                colHint = colHint
            )
        }
    }

    fun assessConfirmRegionAsIsAction(intent: IntentV1): RegionAdmissionV1 {
        val region = intent.regionOrNullCompatV1()
        return if (region == null) {
            RegionAdmissionV1(
                region = null,
                clarification = ActionClarificationSpecV1(
                    missing = listOf("region"),
                    prompt = promptForRegionKind(),
                    kind = ClarifyKind.POSITION
                )
            )
        } else {
            RegionAdmissionV1(region = region)
        }
    }

    fun assessConfirmRegionToDigitsAction(intent: IntentV1): RegionDigitsAdmissionV1 {
        val region = intent.regionOrNullCompatV1()
        if (region == null) {
            return RegionDigitsAdmissionV1(
                region = null,
                digits = null,
                clarification = ActionClarificationSpecV1(
                    missing = listOf("region"),
                    prompt = promptForRegionKind(),
                    kind = ClarifyKind.POSITION
                )
            )
        }

        val sDigits = intent.payload.regionDigits
        val digitsList = intent.payload.digits
        val digits: List<Int>? = when {
            !sDigits.isNullOrBlank() && sDigits.length == 9 ->
                sDigits.map { ch -> (ch.code - '0'.code) }.map { it.coerceIn(0, 9) }

            !digitsList.isNullOrEmpty() && digitsList.size == 9 ->
                digitsList.map { it.coerceIn(0, 9) }

            else -> null
        }

        return if (digits == null) {
            RegionDigitsAdmissionV1(
                region = region,
                digits = null,
                clarification = ActionClarificationSpecV1(
                    missing = listOf("regionDigits"),
                    prompt = promptForRegionDigits(),
                    kind = ClarifyKind.DIGIT
                )
            )
        } else {
            RegionDigitsAdmissionV1(
                region = region,
                digits = digits
            )
        }
    }

    fun assessCandidateAddAction(intent: IntentV1): CandidateDigitsAdmissionV1 =
        assessCandidateDigitsAction(
            intent = intent,
            prompt = promptForCandidateAdd()
        )

    fun assessCandidateRemoveAction(intent: IntentV1): CandidateDigitsAdmissionV1 =
        assessCandidateDigitsAction(
            intent = intent,
            prompt = promptForCandidateRemove()
        )

    fun assessCandidateSetAction(intent: IntentV1): CandidateDigitsAdmissionV1 =
        assessCandidateDigitsAction(
            intent = intent,
            prompt = promptForCandidateSet(),
            allowSingleDigitFallback = false
        )

    private fun assessCellDigitRepairAction(
        s: SudoState,
        intent: IntentV1,
        actionVerb: String
    ): CellDigitAdmissionV1 {
        val idx = mergedRepairCellIndexCompatV1(s, intent)
        val digit = mergedRepairDigitCompatV1(s, intent)
        val (rowHint, colHint) = rowColHintsFromCellIndex(idx)

        val missingCell = idx == null
        val missingDigit = digit == null

        val missing = buildList {
            if (missingCell) {
                if (rowHint == null) add("row")
                if (colHint == null) add("column")
            }
            if (missingDigit) add("digit")
        }

        return if (missing.isNotEmpty()) {
            CellDigitAdmissionV1(
                cellIndex = idx,
                digit = digit,
                rowHint = rowHint,
                colHint = colHint,
                clarification = ActionClarificationSpecV1(
                    missing = missing,
                    prompt = buildRepairClarificationPrompt(
                        actionVerb = actionVerb,
                        rowHint = rowHint,
                        colHint = colHint,
                        needCell = missingCell,
                        needDigit = missingDigit
                    ),
                    kind = if (missingDigit && !missingCell) ClarifyKind.DIGIT else ClarifyKind.POSITION,
                    rowHint = rowHint,
                    colHint = colHint,
                    digitHint = digit?.takeIf { it in 1..9 }
                )
            )
        } else {
            CellDigitAdmissionV1(
                cellIndex = idx,
                digit = digit,
                rowHint = rowHint,
                colHint = colHint
            )
        }
    }

    private fun assessCandidateDigitsAction(
        intent: IntentV1,
        prompt: String,
        allowSingleDigitFallback: Boolean = true
    ): CandidateDigitsAdmissionV1 {
        val idx = intent.cellIndexOrNullCompatV1()
        val digits =
            if (allowSingleDigitFallback) {
                intent.payload.digits ?: (intent.payload.digit?.let { listOf(it) })
            } else {
                intent.payload.digits
            }

        val missing = buildList {
            if (idx == null) add("cell")
            if (digits.isNullOrEmpty()) add("digits")
        }

        return if (missing.isNotEmpty()) {
            CandidateDigitsAdmissionV1(
                cellIndex = idx,
                digits = digits,
                clarification = ActionClarificationSpecV1(
                    missing = missing,
                    prompt = prompt,
                    kind = ClarifyKind.POSITION
                )
            )
        } else {
            CandidateDigitsAdmissionV1(
                cellIndex = idx,
                digits = digits
            )
        }
    }

    private fun buildMissingCellParts(
        rowHint: Int?,
        colHint: Int?
    ): List<String> =
        buildList {
            if (rowHint == null) add("row")
            if (colHint == null) add("column")
        }

    private fun mergedRepairCellIndexCompatV1(
        s: SudoState,
        intent: IntentV1
    ): Int? = intent.cellIndexOrNullCompatV1() ?: cellIndexHintFromPending(s.pending)

    private fun mergedRepairDigitCompatV1(
        s: SudoState,
        intent: IntentV1
    ): Int? = intent.digitOrNullCompatV1() ?: digitHintFromPending(s.pending)

    private fun IntentV1.cellIndexOrNullCompatV1(): Int? =
        this.targets
            .asSequence()
            .mapNotNull { it.cell?.trim() }
            .mapNotNull { cellIndexOf(it) }
            .firstOrNull()

    private fun IntentV1.regionOrNullCompatV1(): RegionRefV1? =
        this.targets
            .asSequence()
            .mapNotNull { it.region }
            .firstOrNull()

    private fun IntentV1.digitOrNullCompatV1(): Int? =
        this.payload.digit ?: this.payload.digits?.firstOrNull()

    // ============================================================
    // Patch Series 4 — detour clarification admission
    // ============================================================

    data class DetourClarificationAdmissionV1(
        val clarification: ActionClarificationSpecV1? = null
    )

    fun assessDetourQuestionAdmission(
        questionClass: DetourQuestionClassV1,
        routingPhase: GridPhase,
        explicitCellRef: String?,
        explicitDigit: Int?,
        explicitRegionRef: String?,
        anchorCellRef: String?,
        techniqueLabel: String?
    ): DetourClarificationAdmissionV1 {
        val clar =
            when (questionClass) {
                DetourQuestionClassV1.PROOF_CHALLENGE ->
                    assessProofChallengeAdmission(
                        routingPhase = routingPhase,
                        explicitCellRef = explicitCellRef,
                        explicitDigit = explicitDigit,
                        explicitRegionRef = explicitRegionRef
                    )

                DetourQuestionClassV1.TARGET_CELL_QUERY ->
                    if (explicitCellRef == null) {
                        ActionClarificationSpecV1(
                            missing = listOf("cell"),
                            prompt = promptForDetourTargetCell(routingPhase),
                            kind = ClarifyKind.POSITION
                        )
                    } else {
                        null
                    }

                DetourQuestionClassV1.NEIGHBOR_CELL_QUERY ->
                    if (explicitCellRef == null && explicitRegionRef == null && anchorCellRef == null) {
                        ActionClarificationSpecV1(
                            missing = listOf("scope"),
                            prompt = promptForDetourNeighborScope(routingPhase),
                            kind = ClarifyKind.POSITION
                        )
                    } else {
                        null
                    }

                DetourQuestionClassV1.CANDIDATE_STATE_QUERY ->
                    if (explicitCellRef == null && explicitRegionRef == null) {
                        ActionClarificationSpecV1(
                            missing = listOf("scope"),
                            prompt = promptForDetourCandidateScope(routingPhase),
                            kind = ClarifyKind.POSITION
                        )
                    } else {
                        null
                    }

                DetourQuestionClassV1.USER_REASONING_CHECK ->
                    if (explicitCellRef == null) {
                        ActionClarificationSpecV1(
                            missing = listOf("cell"),
                            prompt = promptForDetourReasoningCell(routingPhase),
                            kind = ClarifyKind.POSITION
                        )
                    } else {
                        null
                    }

                DetourQuestionClassV1.ALTERNATIVE_TECHNIQUE_QUERY ->
                    if (explicitCellRef == null && explicitRegionRef == null && techniqueLabel == null) {
                        ActionClarificationSpecV1(
                            missing = listOf("scope"),
                            prompt = promptForDetourAlternativeTechniqueScope(routingPhase),
                            kind = ClarifyKind.POSITION
                        )
                    } else {
                        null
                    }

                DetourQuestionClassV1.ROUTE_COMPARISON_QUERY ->
                    if (explicitCellRef == null && explicitRegionRef == null && techniqueLabel == null) {
                        ActionClarificationSpecV1(
                            missing = listOf("scope"),
                            prompt = promptForDetourRouteComparisonScope(routingPhase),
                            kind = ClarifyKind.POSITION
                        )
                    } else {
                        null
                    }

                else -> null
            }

        return DetourClarificationAdmissionV1(clarification = clar)
    }

    fun promptForDetourProofScope(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which cell, row, column, or box do you want me to explain in the current step?"
        } else {
            "Which cell, row, column, or box do you want me to inspect or explain on the board?"
        }

    fun promptForDetourProofDigitInHouse(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which digit in that row, column, or box do you want me to explain in the current step?"
        } else {
            "Which digit in that row, column, or box do you want me to explain on the board?"
        }

    fun promptForDetourTargetCell(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which cell in the current step do you mean?"
        } else {
            "Which cell on the board do you mean?"
        }

    fun promptForDetourNeighborScope(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which nearby cell or region do you want me to inspect in the current step?"
        } else {
            "Which nearby cell or region on the board do you want me to inspect?"
        }

    fun promptForDetourCandidateScope(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which cell, row, column, or box do you want me to inspect in the current solving context?"
        } else {
            "Which cell, row, column, or box do you want me to inspect on the board?"
        }

    fun promptForDetourReasoningCell(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which cell in the current solving step are you proposing something about?"
        } else {
            "Which cell are you proposing something about?"
        }

    fun promptForDetourAlternativeTechniqueScope(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which cell, region, or technique do you want me to compare in the current solving step?"
        } else {
            "Which cell, region, or technique do you want me to compare on the board?"
        }

    fun promptForDetourRouteComparisonScope(
        routingPhase: GridPhase
    ): String =
        if (routingPhase == GridPhase.SOLVING) {
            "Which route, cell, or region do you want me to compare in the current solving context?"
        } else {
            "Which route, cell, or region do you want me to compare on the board?"
        }

    private fun assessProofChallengeAdmission(
        routingPhase: GridPhase,
        explicitCellRef: String?,
        explicitDigit: Int?,
        explicitRegionRef: String?
    ): ActionClarificationSpecV1? {
        // Wave 1 strict rule:
        // A proof challenge must not be admitted on anchor carry alone.
        // This prevents vague prompts like "why is 7" from silently borrowing
        // the prior step target / prior detour anchor and turning into a proof.
        return when {
            explicitCellRef != null ->
                null

            explicitRegionRef != null && explicitDigit in 1..9 ->
                null

            explicitRegionRef != null && explicitDigit == null ->
                ActionClarificationSpecV1(
                    missing = listOf("digit"),
                    prompt = promptForDetourProofDigitInHouse(routingPhase),
                    kind = ClarifyKind.DIGIT
                )

            else ->
                ActionClarificationSpecV1(
                    missing = listOf("scope"),
                    prompt = promptForDetourProofScope(routingPhase),
                    kind = ClarifyKind.POSITION
                )
        }
    }
}

/**
 * High-level clarification family.
 *
 * This is intentionally distinct from the legacy ClarifyKind enum. ClarifyKind
 * remains the low-level pending/app contract already used by the app today.
 * ClarificationFamilyV1 is the new policy-level grouping that later waves can
 * use to centralize reasoning without being boxed in by existing pending names.
 */
enum class ClarificationFamilyV1 {
    POSITION,
    DIGIT,
    HOUSE,
    WORKFLOW,
    REFERENCE,
    INTERPRETATION,
    CONFIRMATION,
    UNKNOWN
}

/**
 * More detailed policy reason for why clarification is being considered.
 */
enum class ClarificationReasonV1 {
    MISSING_CELL,
    MISSING_DIGIT,
    MISSING_HOUSE,
    MISSING_POSITION_PART,
    AMBIGUOUS_REFERENCE,
    INSUFFICIENT_DETOUR_ANCHOR,
    MULTIPLE_COMPETING_ANCHORS,
    INCOMPLETE_WORKFLOW_REQUEST,
    LOW_CONFIDENCE_INTERPRETATION,
    UNKNOWN
}

/**
 * Structured slot inventory for what the user still needs to specify.
 *
 * We keep this explicit instead of scattering stringly-typed "missing" checks
 * throughout SudoConductor.
 */
enum class ClarificationSlotV1 {
    CELL,
    ROW,
    COLUMN,
    DIGIT,
    HOUSE_KIND,
    HOUSE_INDEX,
    TARGET_KIND,
    WORKFLOW_KIND,
    REFERENCE_TARGET,
    YES_NO
}

/**
 * Whether carry-over from prior context is allowed.
 *
 * Future rule of thumb:
 * - STRICT: do not inherit anything; user must restate the missing anchor
 * - BOUNDED: may inherit only one tightly-bounded active anchor
 * - ALLOWED: inheritance is lawful for this clarification family
 */
enum class ClarificationCarryPolicyV1 {
    STRICT,
    BOUNDED,
    ALLOWED
}

/**
 * Whether the utterance has enough grounding to become a full detour / action
 * without clarification.
 *
 * Patch Series 1 only defines the model.
 * The real admission rules arrive in later patch series.
 */
enum class ClarificationAnchorSufficiencyV1 {
    SUFFICIENT,
    INSUFFICIENT,
    UNKNOWN
}

/**
 * Structured anchor summary for detour admission checks.
 */
data class ClarificationAnchorStateV1(
    val cellRef: String? = null,
    val row: Int? = null,
    val col: Int? = null,
    val digit: Int? = null,
    val houseKind: String? = null,
    val houseIndex: Int? = null,
    val source: String? = null,
    val sufficiency: ClarificationAnchorSufficiencyV1 = ClarificationAnchorSufficiencyV1.UNKNOWN
)

/**
 * Future prompt shape for clarification turns.
 *
 * We separate:
 * - the speech text we want the assistant to say
 * - the missing slots it is trying to collect
 * - the policy family / reason behind the prompt
 */
data class ClarificationPromptSpecV1(
    val family: ClarificationFamilyV1,
    val reason: ClarificationReasonV1,
    val missingSlots: List<ClarificationSlotV1>,
    val carryPolicy: ClarificationCarryPolicyV1 = ClarificationCarryPolicyV1.STRICT,
    val prompt: String,
    val rowHint: Int? = null,
    val colHint: Int? = null,
    val digitHint: Int? = null
) {
    fun toPendingKind(): ClarifyKind =
        when (family) {
            ClarificationFamilyV1.POSITION -> ClarifyKind.POSITION
            ClarificationFamilyV1.DIGIT -> ClarifyKind.DIGIT
            ClarificationFamilyV1.HOUSE -> ClarifyKind.WORKFLOW
            ClarificationFamilyV1.WORKFLOW -> ClarifyKind.WORKFLOW
            ClarificationFamilyV1.REFERENCE -> ClarifyKind.POSITION
            ClarificationFamilyV1.INTERPRETATION -> ClarifyKind.WORKFLOW
            ClarificationFamilyV1.CONFIRMATION -> ClarifyKind.YESNO
            ClarificationFamilyV1.UNKNOWN -> ClarifyKind.WORKFLOW
        }

    fun toPending(): Pending.AskClarification =
        Pending.AskClarification(
            kind = toPendingKind(),
            rowHint = rowHint,
            colHint = colHint,
            digitHint = digitHint,
            prompt = prompt
        )
}

/**
 * Policy-level clarification command that later waves can return from the kernel.
 */
data class ClarificationDecisionV1(
    val family: ClarificationFamilyV1,
    val reason: ClarificationReasonV1,
    val promptSpec: ClarificationPromptSpecV1,
    val missingSlots: List<ClarificationSlotV1>,
    val anchorState: ClarificationAnchorStateV1 = ClarificationAnchorStateV1(),
    val sourceIntentId: String? = null,
    val notes: String? = null
) {
    /**
     * Patch Series 1:
     * Minimal bridge into the current UserAgendaItem model.
     * Later patch series may enrich this further.
     */
    fun toUserAgendaClarification(
        id: String,
        createdTurnSeq: Long,
        checkpointRouteId: String? = null,
        checkpointPhase: GridPhase? = null,
        checkpointAppAgendaKind: String? = null,
        checkpointStepId: String? = null,
        checkpointStoryStage: com.contextionary.sudoku.conductor.StoryStage? = null
    ): UserAgendaItem.Clarification {
        val missing =
            if (missingSlots.isNotEmpty()) {
                missingSlots.map { it.name.lowercase() }
            } else {
                emptyList()
            }

        return UserAgendaItem.Clarification(
            id = id,
            intentId = sourceIntentId ?: "clarification:unknown_intent",
            missing = missing,
            askedTurnSeq = null,
            createdTurnSeq = createdTurnSeq,
            prompt = promptSpec.prompt,
            checkpointRouteId = checkpointRouteId,
            checkpointPhase = checkpointPhase,
            checkpointAppAgendaKind = checkpointAppAgendaKind,
            checkpointStepId = checkpointStepId,
            checkpointStoryStage = checkpointStoryStage
        )
    }
}

/**
 * Result of asking "do we need clarification right now?"
 */
sealed class ClarificationAssessmentV1 {
    abstract val reason: String

    data class NotNeeded(
        override val reason: String
    ) : ClarificationAssessmentV1()

    data class Needed(
        val decision: ClarificationDecisionV1,
        override val reason: String
    ) : ClarificationAssessmentV1()
}

/**
 * Result of asking "did the new user utterance resolve the active clarification?"
 */
sealed class ClarificationResolutionAssessmentV1 {
    abstract val reason: String

    data class NotResolved(
        override val reason: String
    ) : ClarificationResolutionAssessmentV1()

    data class PartiallyResolved(
        val recoveredSlots: Map<ClarificationSlotV1, String>,
        override val reason: String
    ) : ClarificationResolutionAssessmentV1()

    data class Resolved(
        val recoveredSlots: Map<ClarificationSlotV1, String>,
        override val reason: String
    ) : ClarificationResolutionAssessmentV1()
}