package com.contextionary.sudoku.conductor

/**
 * Sovereign turn-owner arbiter.
 *
 * Constitutional order:
 *  1) choose one owner for the turn
 *  2) let demand derive from that owner
 *  3) let projection derive from that owner
 *  4) let pending and route-return remain subordinate artifacts
 *
 * This object is the single source of truth for turn authority precedence.
 * Legacy TurnOwnerV1 is compatibility-only and must not become a second sovereign.
 *
 * Must-pass regression hooks:
 *  - DETOUR_DURING_POST_COMMIT_STAYS_USER_DETOUR_OWNER
 *  - ROUTE_JUMP_NEXT_FROM_POST_COMMIT_BECOMES_USER_ROUTE_JUMP_OWNER
 *  - REPAIR_SIGNAL_BEATS_ROUTE_CONTINUATION
 *  - EMPTY_USER_QUEUE_WITH_APP_QUEUE_BECOMES_APP_ROUTE_OWNER
 */
object TurnAuthorityArbiterV1 {

    private fun authorityOwnerHintFromUserAgendaHeadV1(
        head: UserAgendaItem?
    ): TurnAuthorityOwnerV1? =
        when (head) {
            null -> null
            is UserAgendaItem.RouteControl -> TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER
            else -> TurnAuthorityOwnerV1.USER_DETOUR_OWNER
        }

    private fun isRepairPendingForAuthorityV1(
        pending: Pending?
    ): Boolean =
        when (pending) {
            is Pending.ConfirmEdit,
            is Pending.ConfirmCellAsIs,
            is Pending.ConfirmCellToDigit,
            is Pending.ConfirmRegionAsIs,
            is Pending.ConfirmRegionToDigits,
            is Pending.ProvideDigit,
            is Pending.AskCellValue,
            is Pending.ConfirmInterpretation,
            is Pending.AskClarification,
            is Pending.WaitForTap -> true

            else -> false
        }

    private fun isRouteContinuationPendingForAuthorityV1(
        pending: Pending?
    ): Boolean =
        when (pending) {
            is Pending.SolveIntroAction,
            is Pending.AfterResolution,
            is Pending.ApplyHintNow,
            is Pending.ReturnToRoute,
            is Pending.UserAgendaBridge -> true

            else -> false
        }

    private fun resumedPendingKindForAuthorityV1(
        pending: Pending?
    ): String? =
        when (pending) {
            is Pending.ReturnToRoute -> pending.resumePending?.javaClass?.simpleName
            else -> null
        }

    private fun pendingAuthorityStatusForOwnerV1(
        owner: TurnAuthorityOwnerV1,
        pending: Pending?
    ): PendingAuthorityStatusV1 {
        if (pending == null) return PendingAuthorityStatusV1.NONE

        return when (owner) {
            TurnAuthorityOwnerV1.APP_ROUTE_OWNER ->
                when (pending) {
                    is Pending.ReturnToRoute -> PendingAuthorityStatusV1.TRANSFORMED
                    else -> PendingAuthorityStatusV1.ACTIVE
                }

            TurnAuthorityOwnerV1.USER_DETOUR_OWNER ->
                if (isRouteContinuationPendingForAuthorityV1(pending)) {
                    PendingAuthorityStatusV1.SUSPENDED
                } else {
                    PendingAuthorityStatusV1.ACTIVE
                }

            TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER ->
                if (isRouteContinuationPendingForAuthorityV1(pending)) {
                    PendingAuthorityStatusV1.TRANSFORMED
                } else {
                    PendingAuthorityStatusV1.ACTIVE
                }

            TurnAuthorityOwnerV1.REPAIR_OWNER ->
                if (isRouteContinuationPendingForAuthorityV1(pending)) {
                    PendingAuthorityStatusV1.SUBORDINATED
                } else {
                    PendingAuthorityStatusV1.ACTIVE
                }

            TurnAuthorityOwnerV1.NONE ->
                PendingAuthorityStatusV1.NONE
        }
    }

    private fun authorityReasonForOwnerV1(owner: TurnAuthorityOwnerV1): String =
        when (owner) {
            TurnAuthorityOwnerV1.REPAIR_OWNER ->
                "wave2_repair_owner_from_pending_signal"
            TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER ->
                "wave2_route_jump_owner_from_user_route_control"
            TurnAuthorityOwnerV1.USER_DETOUR_OWNER ->
                "wave2_detour_owner_from_user_agenda"
            TurnAuthorityOwnerV1.APP_ROUTE_OWNER ->
                "wave2_app_route_owner_from_app_agenda"
            TurnAuthorityOwnerV1.NONE ->
                "wave2_no_owner"
        }

    fun evidenceSummary(
        decision: TurnAuthorityDecisionV1
    ): List<String> =
        decision.evidence.map { ev ->
            val suffix = ev.detail?.takeIf { it.isNotBlank() }?.let { ":$it" } ?: ""
            ev.kind.name + suffix
        }

    /**
     * Wave 2 — sovereign traffic controller.
     *
     * Precedence:
     *   1) repair owner
     *   2) user route-jump owner
     *   3) user detour owner
     *   4) app route owner
     *   5) none
     *
     * This is the first wave where constitutional authority is chosen explicitly
     * instead of merely mirroring the legacy queue owner.
     */
    fun resolve(
        userAgendaQueue: List<UserAgendaItem>,
        appAgendaQueue: List<AppAgendaItem>,
        canonicalPosition: CanonicalSolvingStepPositionV1?,
        pending: Pending?
    ): TurnAuthorityDecisionV1 {
        val userHead = userAgendaQueue.firstOrNull()
        val appHead = appAgendaQueue.firstOrNull()
        val userOwnerHint = authorityOwnerHintFromUserAgendaHeadV1(userHead)
        val repairRequested = isRepairPendingForAuthorityV1(pending)

        val owner =
            when {
                repairRequested -> TurnAuthorityOwnerV1.REPAIR_OWNER
                userOwnerHint == TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER ->
                    TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER
                userOwnerHint == TurnAuthorityOwnerV1.USER_DETOUR_OWNER ->
                    TurnAuthorityOwnerV1.USER_DETOUR_OWNER
                appAgendaQueue.isNotEmpty() ->
                    TurnAuthorityOwnerV1.APP_ROUTE_OWNER
                else ->
                    TurnAuthorityOwnerV1.NONE
            }

        val evidence = buildList {
            if (userAgendaQueue.isNotEmpty()) {
                add(
                    TurnAuthorityEvidenceV1(
                        kind = TurnAuthorityEvidenceKindV1.USER_QUEUE,
                        detail = userHead?.javaClass?.simpleName
                    )
                )
            }
            if (appAgendaQueue.isNotEmpty()) {
                add(
                    TurnAuthorityEvidenceV1(
                        kind = TurnAuthorityEvidenceKindV1.APP_QUEUE,
                        detail = appHead?.javaClass?.simpleName
                    )
                )
            }
            when (userOwnerHint) {
                TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER ->
                    add(
                        TurnAuthorityEvidenceV1(
                            kind = TurnAuthorityEvidenceKindV1.INTENT_ROUTE_JUMP,
                            detail = (userHead as? UserAgendaItem.RouteControl)?.control?.name
                        )
                    )
                TurnAuthorityOwnerV1.USER_DETOUR_OWNER ->
                    add(
                        TurnAuthorityEvidenceV1(
                            kind = TurnAuthorityEvidenceKindV1.INTENT_DETOUR,
                            detail = userHead?.javaClass?.simpleName
                        )
                    )
                else -> Unit
            }
            if (repairRequested) {
                add(
                    TurnAuthorityEvidenceV1(
                        kind = TurnAuthorityEvidenceKindV1.INTENT_REPAIR,
                        detail = pending?.javaClass?.simpleName
                    )
                )
            }
            canonicalPosition?.let {
                add(
                    TurnAuthorityEvidenceV1(
                        kind = TurnAuthorityEvidenceKindV1.CANONICAL_POSITION,
                        detail = it.kind.name
                    )
                )
            }
            pending?.let {
                add(
                    TurnAuthorityEvidenceV1(
                        kind = TurnAuthorityEvidenceKindV1.PENDING_STATE,
                        detail = it.javaClass.simpleName
                    )
                )
            }
        }

        val pendingAuthorityStatus = pendingAuthorityStatusForOwnerV1(owner, pending)

        return TurnAuthorityDecisionV1(
            owner = owner,
            reason = authorityReasonForOwnerV1(owner),
            evidence = evidence,
            canonicalPositionKind = canonicalPosition?.kind?.name,
            pendingKind = pending?.javaClass?.simpleName,
            appQueueHeadKind = appHead?.javaClass?.simpleName,
            userQueueHeadKind = userHead?.javaClass?.simpleName,
            routeJumpRequested = owner == TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER,
            repairRequested = owner == TurnAuthorityOwnerV1.REPAIR_OWNER,
            detourRequested = owner == TurnAuthorityOwnerV1.USER_DETOUR_OWNER,
            appRouteRequested = owner == TurnAuthorityOwnerV1.APP_ROUTE_OWNER,
            suspendedPendingKind =
                if (
                    pendingAuthorityStatus == PendingAuthorityStatusV1.SUSPENDED ||
                    pendingAuthorityStatus == PendingAuthorityStatusV1.SUBORDINATED
                ) {
                    pending?.javaClass?.simpleName
                } else {
                    null
                },
            pendingAuthorityStatus = pendingAuthorityStatus,
            resumedPendingKind = resumedPendingKindForAuthorityV1(pending),
            routeReturnAllowed =
                owner == TurnAuthorityOwnerV1.USER_DETOUR_OWNER ||
                        owner == TurnAuthorityOwnerV1.REPAIR_OWNER
        )
    }
}