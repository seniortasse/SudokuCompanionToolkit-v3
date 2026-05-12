package com.contextionary.sudoku.conductor.policy

import com.contextionary.sudoku.conductor.DetourProofChallengeLaneV1

/**
 * IntentBinderV1 — deterministic post-processor after Tick-1 parsing.
 *
 * Phase-2 scope (safe + conservative):
 * - Bind YES/NO intents to pending target cell when pending expects YESNO
 * - Fill missing cell target from focus_cell only when Tick-1 explicitly emitted
 *   reference_resolution_mode=FOCUS_CELL
 * - Merge duplicates deterministically
 */
object IntentBinderV1 {

    data class BindDiagnostics(
        val didBindYesNoToPending: Boolean = false,
        val didFillTargetFromFocus: Boolean = false,
        val didMergeDuplicates: Boolean = false,

        // Phase A — assistant-question follow-up seam
        // Note: constitutionally USER_DETOUR intents must never be downgraded into
        // assistant-followup behavior by downstream solving normalization.
        val didBindAssistantQuestionFollowup: Boolean = false,
        val assistantFollowupDisposition: String? = null,

        // Phase 1 — detour routing seam (classification only; diagnostic only)
        val isRouteFollowingIntent: Boolean = false,
        val isRouteControlIntent: Boolean = false,
        val detourQuestionClass: DetourQuestionClassV1? = null,
        val proofChallengeLaneHint: DetourProofChallengeLaneV1? = null,

        // Phase 1 + 2 — solving-road semantic bucket
        val solvingRoadSemantic: SolvingRoadSemanticV2? = null,

        val notes: List<String> = emptyList()
    )

    data class BindResult(
        val envelope: IntentEnvelopeV1,
        val diag: BindDiagnostics
    )

    fun bind(env: IntentEnvelopeV1, ctx: TurnContextV1): BindResult {
        var intents = env.intents.toList()
        val notes = mutableListOf<String>()

        // ------------------------------------------------------------
// Phase 3.C: Non-blocking normalization (type-based ONLY)
// - If CAPABILITY_CHECK / SMALL_TALK / META_APP_QUESTION exist,
//   drop UNKNOWN when it would otherwise pollute the envelope.
// - Keep real grid/workflow intents intact.
// ------------------------------------------------------------
        val nonBlockingTypes = setOf(
            IntentTypeV1.CAPABILITY_CHECK,
            IntentTypeV1.SMALL_TALK,
            IntentTypeV1.META_APP_QUESTION
        )

        val hasNonBlocking = intents.any { it.type in nonBlockingTypes }

        // "Real" = anything that is not UNKNOWN and not one of the non-blocking types.
        val hasRealNonBlockingIntent =
            intents.any { it.type != IntentTypeV1.UNKNOWN && it.type !in nonBlockingTypes }

        if (hasNonBlocking) {
            notes += "non_blocking_intent_present=" +
                    intents.firstOrNull { it.type in nonBlockingTypes }?.type?.name

            // If the only "other" signal is UNKNOWN, drop UNKNOWN.
            if (!hasRealNonBlockingIntent) {
                intents = intents.filterNot { it.type == IntentTypeV1.UNKNOWN }
            }

            // Optional but useful: put non-blocking intents first (stable "top intent")
            val (nb, rest) = intents.partition { it.type in nonBlockingTypes }
            intents = nb + rest
        }

        var didBindYesNoToPending = false
        var didFillTargetFromFocus = false
        var didMergeDuplicates = false

        val pending = ctx.pending


        // ------------------------------------------------------------
        // 0) Retake fork binding (hard override when pending_before=CONFIRM_RETAKE)
        // ------------------------------------------------------------
        val pendingBefore = pending?.pendingBefore?.trim()?.uppercase()
        if (pendingBefore == "CONFIRM_RETAKE") {
            val chooseRetake =
                intents.any { it.type == IntentTypeV1.CHOOSE_RETAKE }

            val chooseKeep =
                intents.any { it.type == IntentTypeV1.CHOOSE_KEEP_SCAN }

            // Only bind this fork from explicit structured intents.
            if (chooseRetake.xor(chooseKeep)) {
                val forcedType =
                    if (chooseRetake) IntentTypeV1.CHOOSE_RETAKE else IntentTypeV1.CHOOSE_KEEP_SCAN

                // Drop confusing "confirm region" intents produced by the model for this fork turn.
                intents = intents.filterNot {
                    it.type == IntentTypeV1.CONFIRM_REGION_AS_IS ||
                            it.type == IntentTypeV1.CONFIRM_REGION_TO_DIGITS
                }

                intents = listOf(
                    IntentV1(
                        id = "binder_forced_retake_fork",
                        type = forcedType,
                        confidence = 0.99,
                        targets = emptyList(),
                        payload = IntentPayloadV1(),
                        missing = emptyList(),
                        evidenceText = "Bound from pending_before=CONFIRM_RETAKE + explicit structured fork intent",
                        addressesUserAgendaId = "pending:confirm_retake"
                    )
                ) + intents

                notes += "Forced ${forcedType.name} because pending_before=CONFIRM_RETAKE and explicit structured fork intent was present"
            }
        }


        // 0B) ✅ SOLVING story CTA normalization (CANONICAL)
        // - Binder collapses legacy solving CTA intents into the new canonical rail.
        // - Golden rule:
        //     * SOLVE_CONTINUE         = advance the app-owned agenda boundary
        //     * SOLVE_ACCEPT_LOCK_IN   = explicit acceptance to commit the current revealed answer
        //     * SOLVE_ACCEPT_NEXT_STEP = explicit acceptance to move from completed step to next step
        //     * SOLVE_PAUSE            = stay on the current step without advancing
        // - Batch B:
        //     obvious forward-driving utterances must bind deterministically in
        //     solving boundary states even if Tick-1 parsing is noisy, UNKNOWN,
        //     or missing the intended enum.
        // ------------------------------------------------------------
        runCatching {
            val pb = pending?.pendingBefore?.trim()?.uppercase() ?: ""
            val canonicalPosition = canonicalSolvingPositionKindUpperV1(ctx)

            val isSolvePreference = (pb == "SOLVE_PREFERENCE")
            val isStepMissing = (pb == "SOLVE_STEP_MISSING")

            val isCanonicalSetup = (canonicalPosition == "SETUP")
            val isCanonicalConfrontation = (canonicalPosition == "CONFRONTATION")
            val isCanonicalResolutionCommit = (canonicalPosition == "RESOLUTION_COMMIT")
            val isCanonicalResolutionPostCommit = (canonicalPosition == "RESOLUTION_POST_COMMIT")

            val blockSolveContinueNormalization =
                shouldBlockSolveContinueNormalizationV1(
                    ctx = ctx,
                    intents = intents
                )

            if (
                isSolvePreference ||
                isStepMissing ||
                isCanonicalSetup ||
                isCanonicalConfrontation ||
                isCanonicalResolutionCommit ||
                isCanonicalResolutionPostCommit
            ) {
                if (blockSolveContinueNormalization) {
                    notes += "SOLVING_CTA_BLOCKED_BY_USER_AGENDA_CLARIFICATION"
                    return@runCatching
                }

                val legacyAndCanonicalSolvingTypes = setOf(
                    IntentTypeV1.UNKNOWN,
                    IntentTypeV1.CONFIRM_YES,
                    IntentTypeV1.SOLVE_CONTINUE,
                    IntentTypeV1.SOLVE_PAUSE,
                    IntentTypeV1.SOLVE_ACCEPT_LOCK_IN,
                    IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP,
                    IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT,
                    IntentTypeV1.REQUEST_EXPLANATION,
                    IntentTypeV1.REQUEST_REASONING_CHECK,
                    IntentTypeV1.ASK_TECHNIQUE_OVERVIEW
                )

                val explicitForwardContinue =
                    intents.any {
                        it.type == IntentTypeV1.SOLVE_CONTINUE ||
                                it.type == IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP ||
                                it.type == IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT
                    }

                val explicitLockInAccept =
                    intents.any { it.type == IntentTypeV1.SOLVE_ACCEPT_LOCK_IN }

                val forcedDeterministicType: IntentTypeV1? = when {
                    isCanonicalResolutionPostCommit && explicitForwardContinue ->
                        IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP

                    isCanonicalResolutionCommit && explicitLockInAccept ->
                        IntentTypeV1.SOLVE_ACCEPT_LOCK_IN

                    isSolvePreference && explicitForwardContinue ->
                        IntentTypeV1.SOLVE_CONTINUE

                    (isCanonicalSetup || isCanonicalConfrontation) && explicitForwardContinue ->
                        IntentTypeV1.SOLVE_CONTINUE

                    isStepMissing && explicitForwardContinue ->
                        IntentTypeV1.SOLVE_CONTINUE

                    else -> null
                }

                val bindingSource =
                    canonicalPosition?.let { "canonical_position=$it" } ?: "pending_before=$pb"

                val bindingAgendaId =
                    canonicalPosition?.let { "canonical:$it" } ?: "pending:$pb"

                if (forcedDeterministicType != null) {
                    intents = intents.filterNot { it.type in legacyAndCanonicalSolvingTypes }

                    intents = listOf(
                        IntentV1(
                            id = "binder_forced_solving_cta",
                            type = forcedDeterministicType,
                            confidence = 0.99,
                            evidenceText = "Deterministically bound from $bindingSource + lexical solving CTA",
                            addressesUserAgendaId = bindingAgendaId
                        )
                    ) + intents

                    notes += "Forced solving CTA to ${forcedDeterministicType.name} for $bindingSource"
                } else {
                    val present = intents.filter { it.type in legacyAndCanonicalSolvingTypes }

                    val preferredType: IntentTypeV1? = when {
                        isCanonicalResolutionPostCommit &&
                                present.any {
                                    it.type == IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP ||
                                            it.type == IntentTypeV1.SOLVE_CONTINUE ||
                                            it.type == IntentTypeV1.CONFIRM_YES
                                } ->
                            IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP

                        isCanonicalResolutionCommit &&
                                present.any {
                                    it.type == IntentTypeV1.SOLVE_ACCEPT_LOCK_IN ||
                                            it.type == IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT
                                } ->
                            IntentTypeV1.SOLVE_ACCEPT_LOCK_IN

                        present.any { it.type == IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP } ->
                            IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP

                        present.any { it.type == IntentTypeV1.SOLVE_ACCEPT_LOCK_IN } ->
                            IntentTypeV1.SOLVE_ACCEPT_LOCK_IN

                        present.any { it.type == IntentTypeV1.SOLVE_CONTINUE } ->
                            IntentTypeV1.SOLVE_CONTINUE

                        (isCanonicalSetup || isCanonicalConfrontation || isSolvePreference || isStepMissing) &&
                                present.any { it.type == IntentTypeV1.CONFIRM_YES } ->
                            IntentTypeV1.SOLVE_CONTINUE

                        present.any { it.type == IntentTypeV1.SOLVE_PAUSE } ->
                            IntentTypeV1.SOLVE_PAUSE

                        present.any { it.type == IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT } ->
                            IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT

                        present.any { it.type == IntentTypeV1.REQUEST_EXPLANATION } ->
                            IntentTypeV1.REQUEST_EXPLANATION

                        present.any { it.type == IntentTypeV1.REQUEST_REASONING_CHECK } ->
                            IntentTypeV1.REQUEST_REASONING_CHECK

                        present.any { it.type == IntentTypeV1.ASK_TECHNIQUE_OVERVIEW } ->
                            IntentTypeV1.ASK_TECHNIQUE_OVERVIEW

                        else -> null
                    }

                    if (preferredType != null) {
                        intents = intents.filterNot { it.type in legacyAndCanonicalSolvingTypes }

                        intents = listOf(
                            IntentV1(
                                id = "binder_normalized_story_cta",
                                type = preferredType,
                                confidence = 0.99,
                                evidenceText = "Normalized solving CTA for $bindingSource",
                                addressesUserAgendaId = bindingAgendaId
                            )
                        ) + intents

                        notes += "Normalized solving CTA to ${preferredType.name} for $bindingSource"
                    } else {
                        notes += "SOLVING_CTA_NO_NORMALIZATION $bindingSource"
                    }
                }
            }
        }.onFailure {
            notes += "SOLVING_CTA_BINDER_ERROR:${it.javaClass.simpleName}"
        }


        val expectsYesNo =
            pending != null &&
                    !pending.expectedAnswerKind.isNullOrBlank() &&
                    (pending.expectedAnswerKind.equals("YESNO", ignoreCase = true) ||
                            pending.expectedAnswerKind.equals("YES_NO", ignoreCase = true))

        // ------------------------------------------------------------
        // 1) YES/NO binding → pending target_cell
        // ------------------------------------------------------------
        if (expectsYesNo && !pending?.targetCell.isNullOrBlank()) {
            val cell = pending!!.targetCell!!
            intents = intents.map { it ->
                if (isYesNoIntent(it.type) && it.targets.isEmpty()) {
                    didBindYesNoToPending = true
                    notes += "Bound ${it.type.name} to pending_target_cell=$cell"
                    it.copy(targets = listOf(IntentTargetV1(cell = cell)))
                } else it
            }
        }


        // ------------------------------------------------------------
        // 1B) Current-step repair binding → pending target cell
        // Use the solving step's target cell for repair / target-cell intents
        // when the LLM has already identified the intent but cell is omitted.
        // ------------------------------------------------------------
        val pendingTargetCell = pending?.targetCell
        val pendingBeforeUpper = pending?.pendingBefore?.trim()?.uppercase()
        val canonicalPositionUpper = canonicalSolvingPositionKindUpperV1(ctx)
        val canUsePendingTargetForRepair =
            !pendingTargetCell.isNullOrBlank() &&
                    (
                            canonicalPositionUpper == "RESOLUTION_COMMIT" ||
                                    canonicalPositionUpper == "RESOLUTION_POST_COMMIT" ||
                                    pendingBeforeUpper == "APPLY_HINT_NOW" ||
                                    pendingBeforeUpper == "AFTER_RESOLUTION"
                            )


        if (canUsePendingTargetForRepair) {
            intents = intents.map { it ->
                val needsCurrentStepTarget =
                    it.targets.isEmpty() && (
                            it.type == IntentTypeV1.EDIT_CELL ||
                                    it.type == IntentTypeV1.CLEAR_CELL ||
                                    it.type == IntentTypeV1.CONFIRM_CELL_AS_IS ||
                                    it.type == IntentTypeV1.CONFIRM_CELL_TO_DIGIT ||
                                    it.type == IntentTypeV1.ASK_CELL_VALUE ||
                                    it.type == IntentTypeV1.ASK_CELL_STATUS
                            )

                if (needsCurrentStepTarget) {
                    didFillTargetFromFocus = true
                    notes += "Filled target for ${it.type.name} from pending_target_cell=$pendingTargetCell (structured current-step target binding)"
                    it.copy(targets = listOf(IntentTargetV1(cell = pendingTargetCell)))
                } else it
            }
        }

        // ------------------------------------------------------------
        // 2) Structured focus-cell binding → focus cell (only if target is missing)
        // Tick 1 is the sole semantic authority for follow-up/deictic meaning.
        // Binder may apply the structured hint, but must not infer it from raw user text.
        // ------------------------------------------------------------
        val focusCell = ctx.focusCell

        if (!focusCell.isNullOrBlank()) {
            intents = intents.map { it ->
                if (shouldFillMissingCellTargetFromFocus(it) && it.targets.isEmpty()) {
                    didFillTargetFromFocus = true
                    notes += "Filled target for ${it.type.name} from focus_cell=$focusCell (reference_resolution_mode=FOCUS_CELL)"
                    it.copy(targets = listOf(IntentTargetV1(cell = focusCell)))
                } else it
            }
        }

        // ------------------------------------------------------------
        // 3) Merge duplicates deterministically (type + targets + payload)
        // ------------------------------------------------------------
        val merged = LinkedHashMap<String, IntentV1>()
        for (it in intents) {
            val k = stableKey(it)
            val prev = merged[k]
            if (prev == null) {
                merged[k] = it
            } else {
                didMergeDuplicates = true

                val keep = if (it.confidence >= prev.confidence) it else prev
                val drop = if (keep === it) prev else it

                merged[k] = keep.copy(
                    missing = unionMissing(keep.missing, drop.missing),
                    id = chooseStableId(keep.id, drop.id)
                )
            }
        }

        val canonicalIntents =
            canonicalizeSolvingIntentsV1(
                intents = merged.values.toList(),
                ctx = ctx,
                notes = notes
            )

        val finalIntents = canonicalIntents

        val isRouteFollowingIntent = finalIntents.any { isRouteFollowingIntentTypeV1(it.type) }
        val isRouteControlIntent = finalIntents.any { isRouteControlIntentTypeV1(it.type) }
        val detourQuestionClass = classifyDetourQuestionClassV1(finalIntents, ctx)
        val proofChallengeLaneHint =
            if (detourQuestionClass == DetourQuestionClassV1.PROOF_CHALLENGE) {
                classifyProofChallengeLaneV1(finalIntents, ctx)
            } else {
                null
            }

        val solvingRoadSemantic =
            classifySolvingRoadSemanticV1(
                intents = finalIntents,
                ctx = ctx,
                isRouteControlIntent = isRouteControlIntent,
                detourQuestionClass = detourQuestionClass
            )

        val didBindAssistantQuestionFollowup =
            isAssistantQuestionFollowupV1(
                intents = finalIntents,
                ctx = ctx,
                detourQuestionClass = detourQuestionClass,
                isRouteControlIntent = isRouteControlIntent
            )

        val assistantFollowupDisposition =
            if (didBindAssistantQuestionFollowup) {
                ctx.awaitedAssistantAnswer?.followupDisposition ?: "APP_ROUTE_FOLLOWUP"
            } else {
                null
            }

        if (isRouteFollowingIntent) notes += "route_following_intent=true"
        if (isRouteControlIntent) notes += "route_control_intent=true"
        if (detourQuestionClass != null) notes += "detour_question_class=${detourQuestionClass.name}"
        if (proofChallengeLaneHint != null) notes += "proof_challenge_lane_hint=${proofChallengeLaneHint.name}"
        if (solvingRoadSemantic != null) notes += "solving_road_semantic=${solvingRoadSemantic.name}"
        if (didBindAssistantQuestionFollowup) {
            notes += "assistant_question_followup=true"
            notes += "assistant_followup_disposition=${assistantFollowupDisposition ?: "APP_ROUTE_FOLLOWUP"}"
        }

        return BindResult(
            envelope = env.copy(intents = finalIntents),
            diag = BindDiagnostics(
                didBindYesNoToPending = didBindYesNoToPending,
                didFillTargetFromFocus = didFillTargetFromFocus,
                didMergeDuplicates = didMergeDuplicates,
                didBindAssistantQuestionFollowup = didBindAssistantQuestionFollowup,
                assistantFollowupDisposition = assistantFollowupDisposition,
                isRouteFollowingIntent = isRouteFollowingIntent,
                isRouteControlIntent = isRouteControlIntent,
                detourQuestionClass = detourQuestionClass,
                proofChallengeLaneHint = proofChallengeLaneHint,
                solvingRoadSemantic = solvingRoadSemantic,
                notes = notes
            )
        )
    }


    private fun isAssistantQuestionFollowupV1(
        intents: List<IntentV1>,
        ctx: TurnContextV1,
        detourQuestionClass: DetourQuestionClassV1?,
        isRouteControlIntent: Boolean
    ): Boolean {
        val awaited = ctx.awaitedAssistantAnswer ?: return false
        if (ctx.userText.isBlank()) return false

        if (isRouteControlIntent) return false
        if (detourQuestionClass != null) return false

        val hasConstitutionalUserDetour =
            intents.any { isUserDetourIntentForAgendaV1(it.type) }

        if (hasConstitutionalUserDetour) return false

        val explicitDetourTypes = setOf(
            IntentTypeV1.CAPABILITY_CHECK,
            IntentTypeV1.SMALL_TALK,
            IntentTypeV1.META_APP_QUESTION,
            IntentTypeV1.REQUEST_EXPLANATION,
            IntentTypeV1.REQUEST_REASONING_CHECK,
            IntentTypeV1.ASK_TECHNIQUE_OVERVIEW,
            IntentTypeV1.SOLVE_STEP_WHY_NOT_DIGIT,
            IntentTypeV1.GRID_MISMATCH_REPORT
        )

        val hasExplicitDetourType = intents.any { it.type in explicitDetourTypes }
        if (hasExplicitDetourType) return false

        return awaited.owner == "APP_ROUTE_OWNER" &&
                awaited.followupDisposition == "APP_ROUTE_FOLLOWUP"
    }

    // -----------------------------
    // Helpers (conservative)
    // -----------------------------

    private fun isYesNoIntent(t: IntentTypeV1): Boolean {
        return t == IntentTypeV1.CONFIRM_YES || t == IntentTypeV1.CONFIRM_NO
    }

    private fun needsCellTarget(i: IntentV1): Boolean {
        return when (i.type) {
            IntentTypeV1.CAPABILITY_CHECK,
            IntentTypeV1.SMALL_TALK,
            IntentTypeV1.META_APP_QUESTION,
            IntentTypeV1.SOLVE_CONTINUE,
            IntentTypeV1.SOLVE_PAUSE,
            IntentTypeV1.SOLVE_ACCEPT_LOCK_IN,
            IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP,
            IntentTypeV1.REQUEST_CURRENT_STAGE_ELABORATION,
            IntentTypeV1.REQUEST_CURRENT_TECHNIQUE_EXPLANATION,
            IntentTypeV1.REQUEST_CURRENT_STAGE_COLLAPSE,
            IntentTypeV1.REQUEST_CURRENT_STAGE_EXAMPLE,
            IntentTypeV1.REQUEST_CURRENT_STAGE_REPEAT,
            IntentTypeV1.REQUEST_CURRENT_STAGE_REPHRASE,
            IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STAGE,
            IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STEP,
            IntentTypeV1.REQUEST_STEP_BACK -> false
            else -> {
                val n = i.type.name
                n.contains("CELL", ignoreCase = true) ||
                        n.contains("EDIT", ignoreCase = true) ||
                        n.contains("CONFIRM", ignoreCase = true)
            }
        }
    }

    private fun shouldFillMissingCellTargetFromFocus(i: IntentV1): Boolean {
        if (!needsCellTarget(i)) return false

        val referenceMode =
            i.referenceResolutionModeCompat()
                ?: i.referenceResolutionMode

        return referenceMode == ReferenceResolutionModeV1.FOCUS_CELL
    }


    private fun isUserAgendaClarificationFollowupV1(ctx: TurnContextV1): Boolean {
        val awaited = ctx.awaitedAssistantAnswer ?: return false

        val ownerUpper = awaited.owner.trim().uppercase()
        val dispositionUpper = awaited.followupDisposition.trim().uppercase()

        if (ownerUpper != "USER_AGENDA_OWNER") return false
        if (dispositionUpper != "USER_AGENDA_DETOUR") return false

        val markers =
            listOf(
                awaited.questionKind,
                awaited.questionKey,
                ctx.lastAssistantQuestionKey,
                ctx.pending?.pendingBefore
            )
                .map { it?.trim()?.uppercase().orEmpty() }

        return markers.any { marker ->
            marker.contains("CLARIFICATION") ||
                    marker.contains("USERAGENDABRIDGE")
        }
    }

    private fun shouldBlockSolveContinueNormalizationV1(
        ctx: TurnContextV1,
        intents: List<IntentV1>
    ): Boolean {
        if (!isUserAgendaClarificationFollowupV1(ctx)) return false
        if (intents.isEmpty()) return false

        val genericRoadTypes = setOf(
            IntentTypeV1.UNKNOWN,
            IntentTypeV1.CONFIRM_YES,
            IntentTypeV1.CONFIRM_NO,
            IntentTypeV1.SOLVE_CONTINUE,
            IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT,
            IntentTypeV1.SOLVE_ACCEPT_LOCK_IN,
            IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP,
            IntentTypeV1.SOLVE_PAUSE,
            IntentTypeV1.SMALL_TALK
        )

        val hasClarificationFill =
            intents.any { it.type == IntentTypeV1.PROVIDE_DIGIT }

        val hasSubstantiveNonRoadContent =
            intents.any { it.type !in genericRoadTypes }

        return hasClarificationFill || hasSubstantiveNonRoadContent
    }


    private fun canonicalSolvingPositionKindUpperV1(ctx: TurnContextV1): String? {
        val canonical = ctx.canonicalSolvingPositionKind?.trim()?.uppercase()
        if (!canonical.isNullOrBlank()) {
            return canonical
        }

        val pb = ctx.pending?.pendingBefore?.trim()?.uppercase()
        return when (pb) {
            "SOLVE_SETUP_ACTION" -> "SETUP"
            "SOLVE_CONFRONTATION_ACTION" -> "CONFRONTATION"
            "APPLY_HINT_NOW" -> "RESOLUTION_COMMIT"
            "AFTER_RESOLUTION" -> "RESOLUTION_POST_COMMIT"

            // Legacy fallback labels kept for compatibility with older traces.
            "SOLVE_INTRO_ACTION" -> "CONFRONTATION"

            else -> null
        }
    }

    private fun isCanonicalSolvingBoundaryPositionV1(ctx: TurnContextV1): Boolean {
        val phaseNow = ctx.phase.trim().uppercase()
        if (phaseNow != "SOLVING") return false

        return when (canonicalSolvingPositionKindUpperV1(ctx)) {
            "SETUP",
            "CONFRONTATION",
            "RESOLUTION_COMMIT",
            "RESOLUTION_POST_COMMIT" -> true

            else -> false
        }
    }

    private fun isCurrentSolvingStageBoundV1(ctx: TurnContextV1): Boolean {
        val phaseNow = ctx.phase.trim().uppercase()
        if (phaseNow != "SOLVING") return false

        if (isCanonicalSolvingBoundaryPositionV1(ctx)) return true

        val pb = ctx.pending?.pendingBefore?.trim()?.uppercase()
        return pb == "SOLVE_PREFERENCE" ||
                pb == "SOLVE_STEP_MISSING" ||
                pb == "SOLVE_SETUP_ACTION" ||
                pb == "SOLVE_CONFRONTATION_ACTION" ||
                pb == "APPLY_HINT_NOW" ||
                pb == "AFTER_RESOLUTION"
    }

    private fun looksCollapsedExplanationRequestV1(userText: String): Boolean {
        return false
    }

    private fun looksExampleRequestV1(userText: String): Boolean {
        return false
    }

    private fun canonicalizeSolvingIntentsV1(
        intents: List<IntentV1>,
        ctx: TurnContextV1,
        notes: MutableList<String>
    ): List<IntentV1> {
        if (intents.isEmpty()) return intents

        val isCurrentStageBound = isCurrentSolvingStageBoundV1(ctx)
        val pendingBeforeUpper = ctx.pending?.pendingBefore?.trim()?.uppercase()
        val canonicalPositionUpper = canonicalSolvingPositionKindUpperV1(ctx)

        return intents.map { intent ->
            val canonicalType =
                when (intent.type) {
                    IntentTypeV1.REQUEST_EXPLANATION -> {
                        val explicitCell =
                            intent.targets.firstOrNull { !it.cell.isNullOrBlank() }?.cell?.trim()

                        val explicitRegion =
                            intent.targets.firstOrNull { it.region != null }?.region

                        val explicitDigit =
                            explicitDigitOrNullV1(intent, ctx.userText)

                        val isSpecificLocalExplanation =
                            explicitDigit != null && (explicitCell != null || explicitRegion != null)

                        if (isSpecificLocalExplanation) {
                            // Do NOT collapse a specifically anchored explanation into stage elaboration.
                            IntentTypeV1.REQUEST_EXPLANATION
                        } else if (isCurrentStageBound) {
                            IntentTypeV1.REQUEST_CURRENT_STAGE_ELABORATION
                        } else {
                            IntentTypeV1.REQUEST_EXPLANATION
                        }
                    }

                    IntentTypeV1.ASK_TECHNIQUE_OVERVIEW -> {
                        if (isCurrentStageBound) {
                            IntentTypeV1.REQUEST_CURRENT_TECHNIQUE_EXPLANATION
                        } else {
                            IntentTypeV1.ASK_TECHNIQUE_OVERVIEW
                        }
                    }

                    IntentTypeV1.CONFIRM_YES -> {
                        when {
                            canonicalPositionUpper == "RESOLUTION_COMMIT" ->
                                IntentTypeV1.SOLVE_ACCEPT_LOCK_IN

                            canonicalPositionUpper == "RESOLUTION_POST_COMMIT" ->
                                IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP

                            pendingBeforeUpper == "APPLY_HINT_NOW" ->
                                IntentTypeV1.SOLVE_ACCEPT_LOCK_IN

                            pendingBeforeUpper == "AFTER_RESOLUTION" ->
                                IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP

                            else ->
                                IntentTypeV1.CONFIRM_YES
                        }
                    }

                    else -> intent.type
                }

            if (canonicalType != intent.type) {
                notes += "canonicalized_${intent.type.name}_to_${canonicalType.name}"
                intent.copy(
                    type = canonicalType,
                    confidence = maxOf(intent.confidence, 0.99),
                    evidenceText = buildString {
                        append(intent.evidenceText)
                        if (isNotBlank()) append(" | ")
                        append("Canonicalized in binder for current solving context")
                    }
                )
            } else {
                intent
            }
        }
    }


    private fun explicitDigitOrNullV1(i: IntentV1, userText: String): Int? {
        // Structured targets only.
        runCatching {
            val targetsJson = i.targets.map { it.toJson() }
            for (tj in targetsJson) {
                if (tj.has("digit") && !tj.isNull("digit")) {
                    val d = tj.optInt("digit", -1)
                    if (d in 1..9) return d
                }
            }
        }

        // Structured payload only.
        runCatching {
            val pj = i.payload.toJson()
            if (pj.has("digit") && !pj.isNull("digit")) {
                val d = pj.optInt("digit", -1)
                if (d in 1..9) return d
            }
        }

        return null
    }

    private fun stableKey(i: IntentV1): String {
        val t = i.type.name
        val targets = i.targets.joinToString(";") { it.toJson().toString() }
        val payload = i.payload.toJson().toString()
        return "$t|$targets|$payload"
    }

    private fun unionMissing(a: List<String>, b: List<String>): List<String> {
        if (a.isEmpty()) return b.distinct()
        if (b.isEmpty()) return a.distinct()
        return (a + b).distinct()
    }

    private fun chooseStableId(a: String, b: String): String {
        if (a.isBlank()) return b
        if (b.isBlank()) return a
        return minOf(a, b)
    }

    // ------------------------------------------------------------
    // Phase 1 — detour routing seam.
    // Route-following is diagnostic only and must not be used to strip
    // constitutionally USER_DETOUR intents of detour ownership.
    // ------------------------------------------------------------

    private fun isRouteFollowingIntentTypeV1(t: IntentTypeV1): Boolean =
        when (t) {
            IntentTypeV1.CONFIRM_YES,
            IntentTypeV1.CONFIRM_NO,
            IntentTypeV1.SOLVE_CONTINUE,
            IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT,
            IntentTypeV1.SOLVE_ACCEPT_LOCK_IN,
            IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP,
            IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STAGE,
            IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STEP,
            IntentTypeV1.REQUEST_STEP_BACK -> true

            else -> false
        }

    private fun isRouteControlIntentTypeV1(t: IntentTypeV1): Boolean =
        isUserRouteJumpIntentForAgendaV1(t)

    private fun isGenericRoadContinueIntentTypeV1(t: IntentTypeV1): Boolean =
        when (t) {
            IntentTypeV1.SOLVE_CONTINUE,
            IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT,
            IntentTypeV1.SOLVE_ACCEPT_LOCK_IN,
            IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP,
            IntentTypeV1.CONFIRM_YES -> true

            else -> false
        }

    private fun looksExplicitRepeatRequestV1(userText: String): Boolean {
        return false
    }

    private fun looksExplicitBackwardRequestV1(userText: String): Boolean {
        return false
    }

    private fun looksExplicitForwardContinueRequestV1(userText: String): Boolean {
        return false
    }

    private fun looksExplicitLockInAcceptanceV1(userText: String): Boolean {
        return false
    }

    private fun classifySolvingRoadSemanticV1(
        intents: List<IntentV1>,
        ctx: TurnContextV1,
        isRouteControlIntent: Boolean,
        detourQuestionClass: DetourQuestionClassV1?
    ): SolvingRoadSemanticV2? {
        val phaseNow = ctx.phase.trim().uppercase()
        if (phaseNow != "SOLVING") return null

        val types = intents.map { it.type }.toSet()

        val hasBackwardIntent =
            types.any {
                it == IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STAGE ||
                        it == IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STEP ||
                        it == IntentTypeV1.REQUEST_STEP_BACK
            }

        val hasRepeatIntent =
            types.any {
                it == IntentTypeV1.REQUEST_CURRENT_STAGE_REPEAT ||
                        it == IntentTypeV1.REQUEST_CURRENT_STAGE_REPHRASE
            }

        val hasStayAndElaborateIntent =
            types.any {
                it == IntentTypeV1.SOLVE_PAUSE ||
                        it == IntentTypeV1.REQUEST_CURRENT_STAGE_ELABORATION ||
                        it == IntentTypeV1.REQUEST_CURRENT_TECHNIQUE_EXPLANATION ||
                        it == IntentTypeV1.REQUEST_CURRENT_STAGE_COLLAPSE ||
                        it == IntentTypeV1.REQUEST_CURRENT_STAGE_EXAMPLE ||
                        it == IntentTypeV1.SOLVE_STEP_WHY_NOT_DIGIT ||
                        it == IntentTypeV1.ASK_WHY_THIS_CELL
            }

        val hasForwardIntent =
            intents.any { isGenericRoadContinueIntentTypeV1(it.type) }

        val isTrueDetourOrRouteControl =
            isRouteControlIntent ||
                    detourQuestionClass == DetourQuestionClassV1.CANDIDATE_STATE_QUERY ||
                    detourQuestionClass == DetourQuestionClassV1.ALTERNATIVE_TECHNIQUE_QUERY ||
                    detourQuestionClass == DetourQuestionClassV1.ROUTE_CONTROL ||
                    detourQuestionClass == DetourQuestionClassV1.NEIGHBOR_CELL_QUERY

        return when {
            isTrueDetourOrRouteControl ->
                SolvingRoadSemanticV2.DETOUR_OR_ROUTE_CONTROL

            hasBackwardIntent ->
                SolvingRoadSemanticV2.GO_BACKWARD

            hasRepeatIntent ->
                SolvingRoadSemanticV2.REPEAT_CURRENT_STAGE

            hasStayAndElaborateIntent ||
                    detourQuestionClass == DetourQuestionClassV1.STEP_CLARIFICATION ||
                    detourQuestionClass == DetourQuestionClassV1.TARGET_CELL_QUERY ||
                    detourQuestionClass == DetourQuestionClassV1.PROOF_CHALLENGE ||
                    detourQuestionClass == DetourQuestionClassV1.USER_REASONING_CHECK ->
                SolvingRoadSemanticV2.STAY_AND_ELABORATE

            hasForwardIntent || intents.isNotEmpty() ->
                SolvingRoadSemanticV2.CONTINUE_FORWARD

            else ->
                null
        }
    }


    private fun classifyProofChallengeLaneV1(
        intents: List<IntentV1>,
        ctx: TurnContextV1
    ): DetourProofChallengeLaneV1 {
        if (intents.isEmpty()) return DetourProofChallengeLaneV1.ELIMINATION_LEGITIMACY

        val types = intents.map { it.type }.toSet()

        if (
            types.any {
                it == IntentTypeV1.CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE ||
                        it == IntentTypeV1.ASK_WHY_THIS_TECHNIQUE_APPLIES_HERE
            }
        ) {
            return DetourProofChallengeLaneV1.TECHNIQUE_LEGITIMACY
        }

        if (types.any { it == IntentTypeV1.CHECK_PROPOSED_ELIMINATION_IN_SCOPE }) {
            return DetourProofChallengeLaneV1.ELIMINATION_LEGITIMACY
        }

        if (types.any { it == IntentTypeV1.ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE }) {
            return DetourProofChallengeLaneV1.HOUSE_BLOCKER
        }

        if (types.any { it == IntentTypeV1.ASK_WHY_DIGIT_IN_CELL }) {
            return DetourProofChallengeLaneV1.CANDIDATE_POSSIBILITY
        }

        if (
            types.any {
                it == IntentTypeV1.ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE ||
                        it == IntentTypeV1.ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT
            }
        ) {
            return DetourProofChallengeLaneV1.FORCEDNESS_OR_UNIQUENESS
        }

        if (
            types.any {
                it == IntentTypeV1.ASK_WHY_THIS_CELL_NOT_OTHER_CELL ||
                        it == IntentTypeV1.ASK_COMPARE_CANDIDATES_BETWEEN_CELLS
            }
        ) {
            return DetourProofChallengeLaneV1.RIVAL_COMPARISON
        }

        if (
            types.any {
                it == IntentTypeV1.REPORT_BUG_OR_WRONG_ASSERTION
            }
        ) {
            return DetourProofChallengeLaneV1.NON_PROOF_OR_NOT_ESTABLISHED
        }

        if (
            types.any {
                it == IntentTypeV1.ASK_WHY_NOT_DIGIT_IN_CELL ||
                        it == IntentTypeV1.SOLVE_STEP_WHY_NOT_DIGIT
            }
        ) {
            return DetourProofChallengeLaneV1.CANDIDATE_IMPOSSIBILITY
        }

        return DetourProofChallengeLaneV1.ELIMINATION_LEGITIMACY
    }



    private fun classifyDetourQuestionClassV1(
        intents: List<IntentV1>,
        ctx: TurnContextV1
    ): DetourQuestionClassV1? {
        if (intents.isEmpty()) return null

        val types = intents.map { it.type }.toSet()
        val pendingTarget = ctx.pending?.targetCell?.trim()

        fun firstExplicitCellTarget(): String? =
            intents.asSequence()
                .flatMap { it.targets.asSequence() }
                .mapNotNull { it.cell?.trim() }
                .firstOrNull()

        val explicitCell = firstExplicitCellTarget()
        val asksAboutPendingTarget =
            !pendingTarget.isNullOrBlank() &&
                    !explicitCell.isNullOrBlank() &&
                    explicitCell.equals(pendingTarget, ignoreCase = true)

        val asksAboutOtherCell =
            !explicitCell.isNullOrBlank() &&
                    (pendingTarget.isNullOrBlank() || !explicitCell.equals(pendingTarget, ignoreCase = true))

        val isExplicitWhyNotDigit =
            types.any {
                it == IntentTypeV1.SOLVE_STEP_WHY_NOT_DIGIT ||
                        it == IntentTypeV1.ASK_WHY_NOT_DIGIT_IN_CELL
            }

        val hasExplicitDigitTarget =
            intents.any { explicitDigitOrNullV1(it, ctx.userText) != null }

        val isCellDigitExplanationChallenge =
            !explicitCell.isNullOrBlank() &&
                    hasExplicitDigitTarget &&
                    types.any {
                        it == IntentTypeV1.REQUEST_EXPLANATION ||
                                it == IntentTypeV1.REPORT_BUG_OR_WRONG_ASSERTION ||
                                it == IntentTypeV1.ASK_WHY_DIGIT_IN_CELL
                    }

        return when {
            types.any {
                it == IntentTypeV1.REQUEST_REASONING_CHECK ||
                        it == IntentTypeV1.CHECK_PROPOSED_DIGIT_IN_CELL ||
                        it == IntentTypeV1.CHECK_PROPOSED_CANDIDATE_SET_IN_CELL ||
                        it == IntentTypeV1.CHECK_PROPOSED_ELIMINATION_IN_SCOPE ||
                        it == IntentTypeV1.CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE ||
                        it == IntentTypeV1.CHECK_PROPOSED_ROUTE_EQUIVALENCE
            } ->
                DetourQuestionClassV1.USER_REASONING_CHECK

            types.any {
                it == IntentTypeV1.ASK_CELL_CANDIDATES ||
                        it == IntentTypeV1.ASK_CELL_CANDIDATES_EXACT ||
                        it == IntentTypeV1.ASK_CANDIDATE_COUNT_CELL ||
                        it == IntentTypeV1.ASK_CELL_CANDIDATE_COUNT_EXACT ||
                        it == IntentTypeV1.ASK_CELLS_WITH_N_CANDIDATES ||
                        it == IntentTypeV1.ASK_HOUSE_CANDIDATE_MAP ||
                        it == IntentTypeV1.ASK_HOUSE_CANDIDATE_MAP_EXACT ||
                        it == IntentTypeV1.ASK_CANDIDATE_FREQUENCY ||
                        it == IntentTypeV1.ASK_CANDIDATES_OVERVIEW ||
                        it == IntentTypeV1.ASK_CANDIDATES_CELL_OVERVIEW ||
                        it == IntentTypeV1.ASK_CANDIDATES_DISTRIBUTION ||
                        it == IntentTypeV1.ASK_DIGIT_LOCATIONS
            } ->
                DetourQuestionClassV1.CANDIDATE_STATE_QUERY

            types.any { it == IntentTypeV1.ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT } &&
                    hasExplicitDigitTarget &&
                    intents.any { intent -> intent.targets.any { it.region != null } } ->
                DetourQuestionClassV1.PROOF_CHALLENGE

            types.any { it == IntentTypeV1.ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT } ->
                DetourQuestionClassV1.CANDIDATE_STATE_QUERY

            types.any { it == IntentTypeV1.ASK_COMPARE_CANDIDATES_BETWEEN_CELLS } ||
                    asksAboutOtherCell ->
                DetourQuestionClassV1.NEIGHBOR_CELL_QUERY

            isExplicitWhyNotDigit ||
                    isCellDigitExplanationChallenge ||
                    types.any {
                        it == IntentTypeV1.ASK_WHY_DIGIT_IN_CELL ||
                                it == IntentTypeV1.ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE ||
                                it == IntentTypeV1.REPORT_BUG_OR_WRONG_ASSERTION
                    } ->
                DetourQuestionClassV1.PROOF_CHALLENGE

            types.any {
                it == IntentTypeV1.ASK_ALTERNATIVE_TECHNIQUE_FOR_CURRENT_SPOT ||
                        it == IntentTypeV1.ASK_TECHNIQUES_NEEDED ||
                        it == IntentTypeV1.ASK_SOLVING_OVERVIEW ||
                        it == IntentTypeV1.ASK_TECHNIQUE_OVERVIEW ||
                        it == IntentTypeV1.ASK_STUCK_HELP ||
                        it == IntentTypeV1.ASK_ADVANCED_PATTERN_HELP ||
                        it == IntentTypeV1.ASK_HIDDEN_SINGLE_LOCATIONS ||
                        it == IntentTypeV1.ASK_NAKED_SINGLE_LOCATIONS ||
                        it == IntentTypeV1.ASK_NAKED_PAIR_LOCATIONS ||
                        it == IntentTypeV1.ASK_HIDDEN_PAIR_LOCATIONS ||
                        it == IntentTypeV1.ASK_POINTING_PAIR_TRIPLE ||
                        it == IntentTypeV1.ASK_BOX_LINE_REDUCTION ||
                        it == IntentTypeV1.ASK_XWING_CANDIDATE
            } ->
                DetourQuestionClassV1.ALTERNATIVE_TECHNIQUE_QUERY

            types.any {
                it == IntentTypeV1.ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE
            } ->
                DetourQuestionClassV1.PROOF_CHALLENGE

            types.any {
                it == IntentTypeV1.ASK_WHY_THIS_CELL ||
                        it == IntentTypeV1.ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT ||
                        it == IntentTypeV1.ASK_WHY_THIS_CELL_NOT_OTHER_CELL
            } ->
                if (hasExplicitDigitTarget || !explicitCell.isNullOrBlank()) {
                    DetourQuestionClassV1.PROOF_CHALLENGE
                } else {
                    DetourQuestionClassV1.TARGET_CELL_QUERY
                }

            asksAboutPendingTarget ->
                DetourQuestionClassV1.TARGET_CELL_QUERY

            types.any {
                it == IntentTypeV1.ASK_OTHER_LOCAL_MOVE_EXISTS
            } ->
                DetourQuestionClassV1.NEIGHBOR_CELL_QUERY

            types.any {
                it == IntentTypeV1.ASK_COMPARE_CURRENT_ROUTE_WITH_ALTERNATIVE_ROUTE
            } ->
                DetourQuestionClassV1.ROUTE_COMPARISON_QUERY

            types.any {
                it == IntentTypeV1.REQUEST_EXPLANATION ||
                        it == IntentTypeV1.ASK_TECHNIQUE_OVERVIEW
            } &&
                    !types.any {
                        it == IntentTypeV1.REQUEST_CURRENT_STAGE_ELABORATION ||
                                it == IntentTypeV1.REQUEST_CURRENT_TECHNIQUE_EXPLANATION ||
                                it == IntentTypeV1.REQUEST_CURRENT_STAGE_COLLAPSE ||
                                it == IntentTypeV1.REQUEST_CURRENT_STAGE_EXAMPLE ||
                                it == IntentTypeV1.REQUEST_CURRENT_STAGE_REPEAT ||
                                it == IntentTypeV1.REQUEST_CURRENT_STAGE_REPHRASE
                    } ->
                DetourQuestionClassV1.STEP_CLARIFICATION

            types.any {
                it == IntentTypeV1.CAPABILITY_CHECK ||
                        it == IntentTypeV1.META_APP_QUESTION
            } ->
                DetourQuestionClassV1.GENERAL_QUESTION

            types.any { isRouteControlIntentTypeV1(it) } ->
                DetourQuestionClassV1.ROUTE_CONTROL

            else -> null
        }
    }

}