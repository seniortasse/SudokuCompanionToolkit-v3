package com.contextionary.sudoku.conductor

/**
 * Explicit checkpoint / boundary controller.
 *
 * Route truth rules:
 *  - detours pause checkpoints
 *  - route-jumps deliberately re-enter or release checkpoints
 *  - repair blocks false advancement
 *  - next-step release must be explicit, never implied by tone alone
 *
 * Must-pass regression hooks:
 *  - DETOUR_PAUSES_CURRENT_CHECKPOINT
 *  - ROUTE_JUMP_CAN_REENTER_TARGET_CHECKPOINT
 *  - NEXT_FROM_RESOLUTION_POST_COMMIT_RELEASES_TO_NEXT_STEP
 *  - REPAIR_BLOCKS_CHECKPOINT_ADVANCE
 */
object RouteBoundaryControllerV1 {

    private fun isNextLikeRouteJumpV1(head: UserAgendaItem?): Boolean {
        val controlName = (head as? UserAgendaItem.RouteControl)?.control?.name ?: return false
        return controlName.contains("NEXT", ignoreCase = true) ||
                controlName.contains("CONTINUE", ignoreCase = true) ||
                controlName.contains("FORWARD", ignoreCase = true)
    }

    fun resolve(
        authorityDecision: TurnAuthorityDecisionV1,
        canonicalPosition: CanonicalSolvingStepPositionV1?,
        pending: Pending?,
        userAgendaQueue: List<UserAgendaItem>
    ): RouteBoundaryDecisionV1 {
        val fromCheckpointKind = canonicalPosition?.kind?.name
        val userHead = userAgendaQueue.firstOrNull()

        val status =
            when {
                authorityDecision.owner == TurnAuthorityOwnerV1.REPAIR_OWNER ->
                    RouteBoundaryStatusV1.BLOCKED_BY_REPAIR

                authorityDecision.owner == TurnAuthorityOwnerV1.USER_DETOUR_OWNER &&
                        canonicalPosition != null ->
                    RouteBoundaryStatusV1.PAUSED_AT_CHECKPOINT

                authorityDecision.owner == TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER &&
                        canonicalPosition?.kind == CanonicalSolvingPositionKindV1.RESOLUTION_POST_COMMIT &&
                        isNextLikeRouteJumpV1(userHead) ->
                    RouteBoundaryStatusV1.RELEASED_TO_NEXT_STEP

                authorityDecision.owner == TurnAuthorityOwnerV1.USER_ROUTE_JUMP_OWNER &&
                        canonicalPosition != null ->
                    RouteBoundaryStatusV1.REENTER_TARGET_CHECKPOINT

                authorityDecision.owner == TurnAuthorityOwnerV1.APP_ROUTE_OWNER &&
                        canonicalPosition?.kind == CanonicalSolvingPositionKindV1.RESOLUTION_POST_COMMIT &&
                        pending !is Pending.ReturnToRoute ->
                    RouteBoundaryStatusV1.STAY_ON_CURRENT_CHECKPOINT

                authorityDecision.owner == TurnAuthorityOwnerV1.APP_ROUTE_OWNER &&
                        canonicalPosition != null ->
                    RouteBoundaryStatusV1.STAY_ON_CURRENT_CHECKPOINT

                else ->
                    RouteBoundaryStatusV1.NONE
            }

        val toCheckpointKind =
            when (status) {
                RouteBoundaryStatusV1.RELEASED_TO_NEXT_STEP ->
                    CanonicalSolvingPositionKindV1.SETUP.name

                RouteBoundaryStatusV1.REENTER_TARGET_CHECKPOINT,
                RouteBoundaryStatusV1.PAUSED_AT_CHECKPOINT,
                RouteBoundaryStatusV1.STAY_ON_CURRENT_CHECKPOINT,
                RouteBoundaryStatusV1.BLOCKED_BY_REPAIR ->
                    fromCheckpointKind

                RouteBoundaryStatusV1.NONE ->
                    null
            }

        val reason =
            when (status) {
                RouteBoundaryStatusV1.RELEASED_TO_NEXT_STEP ->
                    "wave7_release_post_commit_to_next_step"
                RouteBoundaryStatusV1.REENTER_TARGET_CHECKPOINT ->
                    "wave7_route_jump_reenters_target_checkpoint"
                RouteBoundaryStatusV1.PAUSED_AT_CHECKPOINT ->
                    "wave7_detour_pauses_current_checkpoint"
                RouteBoundaryStatusV1.BLOCKED_BY_REPAIR ->
                    "wave7_repair_blocks_checkpoint_advance"
                RouteBoundaryStatusV1.STAY_ON_CURRENT_CHECKPOINT ->
                    "wave7_stay_on_current_checkpoint"
                RouteBoundaryStatusV1.NONE ->
                    "wave7_no_boundary_transition"
            }

        if (status != RouteBoundaryStatusV1.NONE) {
            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                mapOf(
                    "type" to "WAVE_A_ROUTE_BOUNDARY_HOOK_V1",
                    "status" to status.name,
                    "reason" to reason,
                    "from_checkpoint_kind" to fromCheckpointKind,
                    "to_checkpoint_kind" to toCheckpointKind,
                    "route_paused" to (status == RouteBoundaryStatusV1.PAUSED_AT_CHECKPOINT),
                    "advance_allowed" to (status == RouteBoundaryStatusV1.RELEASED_TO_NEXT_STEP)
                )
            )
        }

        return RouteBoundaryDecisionV1(
            status = status,
            reason = reason,
            fromCheckpointKind = fromCheckpointKind,
            toCheckpointKind = toCheckpointKind,
            routePaused = status == RouteBoundaryStatusV1.PAUSED_AT_CHECKPOINT,
            advanceAllowed = status == RouteBoundaryStatusV1.RELEASED_TO_NEXT_STEP
        )
    }
}