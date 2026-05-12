package com.contextionary.sudoku.conductor.policy

import org.json.JSONObject

/**
 * Phase 6 — contract-driven reply assembly planner.
 *
 * This is the first point where the new architecture actually computes:
 * - which prompt modules are selected
 * - which supply channels are selected
 * - projected channel payloads
 *
 * Important:
 * Phase 6 still does NOT force production payload shaping globally.
 * It produces the plan and projected slices so Phase 7 can switch demand
 * categories one by one in a controlled rollout.
 */
object ReplyAssemblyPlannerV1 {

    private const val ENABLE_SOLVING_SETUP_PACKET_V1 = true
    private const val ENABLE_SOLVING_CONFRONTATION_PACKET_V1 = true
    private const val ENABLE_SOLVING_RESOLUTION_PACKET_V1 = true

    data class ProjectedChannelPayloadV1(
        val channel: ReplySupplyChannelV1,
        val payload: JSONObject
    )

    data class PlanResultV1(
        val plan: ReplyAssemblyPlanV1,
        val projectedChannels: List<ProjectedChannelPayloadV1>
    )

    private fun shouldSelectSolvingContinuityShort(replyRequest: ReplyRequestV1): Boolean {
        val pendingBefore = replyRequest.turn.pendingBefore.orEmpty()
        val pendingAfter = replyRequest.turn.pendingAfter.orEmpty()

        val hasRouteReturnPacket =
            replyRequest.facts.any { it.type == FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1 }

        val hasHandoverNote =
            replyRequest.facts.any { it.type == FactBundleV1.Type.HANDOVER_NOTE_V1 }

        val looksLikeDetourBoundary =
            pendingBefore.contains("UserAgendaBridge", ignoreCase = true) ||
                    pendingAfter.contains("UserAgendaBridge", ignoreCase = true) ||
                    pendingBefore.contains("Return", ignoreCase = true) ||
                    pendingAfter.contains("Return", ignoreCase = true)

        return hasRouteReturnPacket || hasHandoverNote || looksLikeDetourBoundary
    }



    private fun shouldSelectPersonalizationMini(
        demand: ReplyDemandResolutionV1
    ): Boolean {
        return when (demand.category) {
            ReplyDemandCategoryV1.ONBOARDING_OPENING,
            ReplyDemandCategoryV1.CONFIRM_STATUS_SUMMARY,
            ReplyDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE,
            ReplyDemandCategoryV1.CONFIRM_FINALIZE_GATE,
            ReplyDemandCategoryV1.CONFIRM_RETAKE_GATE,
            ReplyDemandCategoryV1.CONFIRM_MISMATCH_GATE,
            ReplyDemandCategoryV1.CONFIRM_CONFLICT_GATE,
            ReplyDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE,
            ReplyDemandCategoryV1.FREE_TALK_NON_GRID,
            ReplyDemandCategoryV1.SMALL_TALK_BRIDGE,
            ReplyDemandCategoryV1.SOLVING_SETUP,
            ReplyDemandCategoryV1.SOLVING_CONFRONTATION,
            ReplyDemandCategoryV1.SOLVING_RESOLUTION,
            ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE,
            ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY,
            ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY,
            ReplyDemandCategoryV1.DETOUR_REASONING_CHECK,
            ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE,
            ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH,
            ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON,
            ReplyDemandCategoryV1.REPAIR_CONTRADICTION,
            ReplyDemandCategoryV1.FREE_TALK_IN_GRID_SESSION,
            ReplyDemandCategoryV1.RECOVERY_REPLY -> true

            else -> false
        }
    }

    private fun isPacketCenteredSolvingDetourDemandCategory(
        category: ReplyDemandCategoryV1
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


    private fun doctrineModulesForProofChallengePacket(
        projectedChannels: List<ProjectedChannelPayloadV1>
    ): List<PromptModuleV1> {
        val packet =
            projectedChannels
                .firstOrNull { it.channel == ReplySupplyChannelV1.DETOUR_MOVE_PROOF_PACKET }
                ?.payload
                ?: return emptyList()

        val doctrineId =
            packet.optJSONObject("doctrine")
                ?.optString("id", null)
                ?.trim()
                ?.takeIf { it.isNotEmpty() }

        val archetypeId =
            packet.optJSONObject("narrative_archetype")
                ?.optString("id", null)
                ?.trim()
                ?.takeIf { it.isNotEmpty() }

        val geometryKind =
            packet.optJSONObject("local_proof_geometry")
                ?.optString("geometry_kind", null)
                ?.trim()
                ?.takeIf { it.isNotEmpty() }

        val effectiveDoctrineId =
            doctrineId?.lowercase()
                ?: when (archetypeId?.trim()?.uppercase()) {
                    "LOCAL_CONTRADICTION_SPOTLIGHT" -> "contradiction_spotlight_v1"
                    "LOCAL_PERMISSIBILITY_SCAN" -> "local_permissibility_scan_v1"
                    "SURVIVOR_LADDER" -> "survivor_ladder_v1"
                    "CONTRAST_DUEL" -> "contrast_duel_v1"
                    "PATTERN_LEGITIMACY_CHECK" -> "pattern_legitimacy_v1"
                    "HONEST_INSUFFICIENCY_ANSWER" -> "honest_insufficiency_v1"
                    else -> when (geometryKind?.trim()?.uppercase()) {
                        "CELL_THREE_HOUSE_UNIVERSE" -> "local_permissibility_scan_v1"
                        "HOUSE_DIGIT_SEAT_MAP" -> "survivor_ladder_v1"
                        "RIVAL_COMPARISON_FRAME" -> "contrast_duel_v1"
                        "PATTERN_STRUCTURE_FRAME" -> "pattern_legitimacy_v1"
                        else -> null
                    }
                }

        return when (effectiveDoctrineId) {
            "contradiction_spotlight_v1" ->
                listOf(PromptModuleV1.DETOUR_PROOF_CONTRADICTION_SPOTLIGHT_RULES)

            "local_permissibility_scan_v1" ->
                listOf(PromptModuleV1.DETOUR_PROOF_LOCAL_PERMISSIBILITY_SCAN_RULES)

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
    }



    fun plan(
        demand: ReplyDemandResolutionV1,
        replyRequest: ReplyRequestV1
    ): PlanResultV1 {
        val contract = ReplyDemandContractsV1.forDemand(demand)

        val baseSelectedPromptModules = contract.requiredPromptModules
            .toList()
            .sortedBy { it.name }

        val selectedChannels =
            when (demand.category) {
                ReplyDemandCategoryV1.SOLVING_SETUP,
                ReplyDemandCategoryV1.SOLVING_CONFRONTATION,
                ReplyDemandCategoryV1.SOLVING_RESOLUTION ->
                    linkedSetOf<ReplySupplyChannelV1>().apply {
                        addAll(contract.requiredChannels)
                        if (shouldSelectSolvingContinuityShort(replyRequest)) {
                            add(ReplySupplyChannelV1.CONTINUITY_SHORT)
                        }
                        if (shouldSelectPersonalizationMini(demand) &&
                            !contract.forbiddenChannels.contains(ReplySupplyChannelV1.PERSONALIZATION_MINI)
                        ) {
                            add(ReplySupplyChannelV1.PERSONALIZATION_MINI)
                        }
                    }.toList().sortedBy { it.name }

                else ->
                    if (isPacketCenteredSolvingDetourDemandCategory(demand.category)) {
                        linkedSetOf<ReplySupplyChannelV1>().apply {
                            addAll(contract.requiredChannels)
                            if (shouldSelectSolvingContinuityShort(replyRequest) &&
                                !contract.forbiddenChannels.contains(ReplySupplyChannelV1.CONTINUITY_SHORT)
                            ) {
                                add(ReplySupplyChannelV1.CONTINUITY_SHORT)
                            }
                            if (shouldSelectPersonalizationMini(demand) &&
                                !contract.forbiddenChannels.contains(ReplySupplyChannelV1.PERSONALIZATION_MINI)
                            ) {
                                add(ReplySupplyChannelV1.PERSONALIZATION_MINI)
                            }
                        }.toList().sortedBy { it.name }
                    } else {
                        linkedSetOf<ReplySupplyChannelV1>().apply {
                            addAll(contract.requiredChannels)
                            addAll(contract.optionalChannels)
                            if (shouldSelectPersonalizationMini(demand) &&
                                !contract.forbiddenChannels.contains(ReplySupplyChannelV1.PERSONALIZATION_MINI)
                            ) {
                                add(ReplySupplyChannelV1.PERSONALIZATION_MINI)
                            }
                        }.toList().sortedBy { it.name }
                    }
            }

        val projected = selectedChannels.map { channel ->
            ProjectedChannelPayloadV1(
                channel = channel,
                payload = ReplySupplyProjectorsV1.projectChannel(replyRequest, channel)
            )
        }

        val doctrineModules =
            if (demand.category == ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE) {
                // Series-C P3:
                // Proof-challenge doctrine modules are now full narrative law:
                // they govern opening, spotlight, proof motion, bounded landing,
                // and closure shape for the local proof story.
                doctrineModulesForProofChallengePacket(projected)
            } else {
                emptyList()
            }

        val selectedPromptModules =
            linkedSetOf<PromptModuleV1>().apply {
                addAll(baseSelectedPromptModules)
                addAll(doctrineModules)
            }.toList().sortedBy { it.name }

        if (demand.category == ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE) {
            runCatching {
                com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                    tag = "DETOUR_PROOF_DOCTRINE_MODULES_SELECTED_V1",
                    data = mapOf(
                        "demand_category" to demand.category.name,
                        "doctrine_modules" to doctrineModules.joinToString(",") { it.name },
                        "selected_prompt_modules" to selectedPromptModules.joinToString(",") { it.name }
                    )
                )
            }
        }



        val rolloutMode =
            when (demand.category) {
                ReplyDemandCategoryV1.ONBOARDING_OPENING ->
                    "onboarding_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_STATUS_SUMMARY ->
                    "confirm_status_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE ->
                    "confirm_exact_match_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_FINALIZE_GATE ->
                    "confirm_finalize_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_RETAKE_GATE ->
                    "confirm_retake_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_MISMATCH_GATE ->
                    "confirm_mismatch_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_CONFLICT_GATE ->
                    "confirm_conflict_projected_prompt_v1"

                ReplyDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE ->
                    "confirm_not_unique_projected_prompt_v1"

                ReplyDemandCategoryV1.FREE_TALK_IN_GRID_SESSION ->
                    "meta_state_answer_transitional_alias_v1"

                ReplyDemandCategoryV1.PENDING_CLARIFICATION ->
                    "pending_clarification_projected_prompt_v1"

                ReplyDemandCategoryV1.PENDING_CELL_CONFIRM_AS_IS ->
                    "pending_cell_confirm_as_is_projected_prompt_v1"

                ReplyDemandCategoryV1.PENDING_CELL_CONFIRM_TO_DIGIT ->
                    "pending_cell_confirm_to_digit_projected_prompt_v1"

                ReplyDemandCategoryV1.PENDING_REGION_CONFIRM_AS_IS ->
                    "pending_region_confirm_as_is_projected_prompt_v1"

                ReplyDemandCategoryV1.PENDING_REGION_CONFIRM_TO_DIGITS ->
                    "pending_region_confirm_to_digits_projected_prompt_v1"

                ReplyDemandCategoryV1.PENDING_DIGIT_PROVIDE ->
                    "pending_digit_provide_projected_prompt_v1"

                ReplyDemandCategoryV1.PENDING_INTERPRETATION_CONFIRM ->
                    "pending_interpretation_confirm_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_VALIDATION_ANSWER ->
                    "grid_validation_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_CANDIDATE_ANSWER ->
                    "grid_candidate_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_OCR_TRUST_ANSWER ->
                    "grid_ocr_trust_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_CONTENTS_ANSWER ->
                    "grid_contents_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_CHANGELOG_ANSWER ->
                    "grid_changelog_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_EDIT_EXECUTION ->
                    "grid_edit_execution_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_CLEAR_EXECUTION ->
                    "grid_clear_execution_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_SWAP_EXECUTION ->
                    "grid_swap_execution_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_BATCH_EDIT_EXECUTION ->
                    "grid_batch_edit_execution_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_UNDO_REDO_EXECUTION ->
                    "grid_undo_redo_execution_projected_prompt_v1"

                ReplyDemandCategoryV1.GRID_LOCK_GIVENS_EXECUTION ->
                    "grid_lock_givens_execution_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_STAGE_ELABORATION ->
                    "solving_stage_elaboration_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_STAGE_REPEAT ->
                    "solving_stage_repeat_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_STAGE_REPHRASE ->
                    "solving_stage_rephrase_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_GO_BACKWARD ->
                    "solving_go_backward_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_STEP_REVEAL ->
                    "solving_step_reveal_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_ROUTE_CONTROL ->
                    "solving_route_control_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE ->
                    "detour_proof_challenge_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY ->
                    "detour_target_cell_query_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY ->
                    "detour_neighbor_cell_query_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_REASONING_CHECK ->
                    "detour_reasoning_check_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE ->
                    "detour_alternative_technique_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH ->
                    "detour_local_move_search_projected_prompt_v1"

                ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON ->
                    "detour_route_comparison_projected_prompt_v1"

                ReplyDemandCategoryV1.PREFERENCE_CHANGE ->
                    "preference_change_projected_prompt_v1"

                ReplyDemandCategoryV1.MODE_CHANGE ->
                    "mode_change_projected_prompt_v1"

                ReplyDemandCategoryV1.ASSISTANT_PAUSE_RESUME ->
                    "assistant_pause_resume_projected_prompt_v1"

                ReplyDemandCategoryV1.VALIDATE_ONLY_OR_SOLVE_ONLY ->
                    "validate_only_or_solve_only_projected_prompt_v1"

                ReplyDemandCategoryV1.FOCUS_REDIRECT ->
                    "focus_redirect_projected_prompt_v1"

                ReplyDemandCategoryV1.HINT_POLICY_CHANGE ->
                    "hint_policy_change_projected_prompt_v1"

                ReplyDemandCategoryV1.META_STATE_ANSWER ->
                    "meta_state_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.CAPABILITY_ANSWER ->
                    "capability_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.GLOSSARY_ANSWER ->
                    "glossary_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.UI_HELP_ANSWER ->
                    "ui_help_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.COORDINATE_HELP_ANSWER ->
                    "coordinate_help_answer_projected_prompt_v1"

                ReplyDemandCategoryV1.FREE_TALK_NON_GRID ->
                    "free_talk_non_grid_projected_prompt_v1"

                ReplyDemandCategoryV1.SMALL_TALK_BRIDGE ->
                    "small_talk_bridge_projected_prompt_v1"

                ReplyDemandCategoryV1.SOLVING_SETUP ->
                    if (ENABLE_SOLVING_SETUP_PACKET_V1) {
                        "solving_setup_packet_v1"
                    } else {
                        "legacy_payload"
                    }

                ReplyDemandCategoryV1.SOLVING_CONFRONTATION ->
                    if (ENABLE_SOLVING_CONFRONTATION_PACKET_V1) {
                        "solving_confrontation_packet_v1"
                    } else {
                        "solving_confrontation_projected_prompt_and_body_v1"
                    }

                ReplyDemandCategoryV1.SOLVING_RESOLUTION ->
                    if (ENABLE_SOLVING_RESOLUTION_PACKET_V1) {
                        "solving_resolution_packet_v1"
                    } else {
                        "solving_resolution_projected_prompt_and_body_v1"
                    }

                ReplyDemandCategoryV1.REPAIR_CONTRADICTION ->
                    "repair_projected_prompt_and_body_v1"

                else ->
                    "legacy_payload"
            }

        val canonicalStoryMini = replyRequest.turn.story?.let { story ->
            listOfNotNull(
                story.canonicalPositionKind?.let { "pos=$it" },
                story.canonicalHeadKind?.let { "head=$it" },
                story.canonicalPendingKind?.let { "pending=$it" }
            ).joinToString(" ")
        }?.takeIf { it.isNotBlank() }

        val baseNotes =
            when (demand.category) {
                ReplyDemandCategoryV1.ONBOARDING_OPENING ->
                    "Onboarding opening now uses slim prompt composition; body shaping still legacy."

                ReplyDemandCategoryV1.CONFIRM_STATUS_SUMMARY ->
                    "Wave-1 B1 active: confirm-status summary now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE ->
                    "Wave-1 B2 active: exact-match confirmation now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRM_FINALIZE_GATE ->
                    "Wave-1 B7 active: finalize/start-solving handoff now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRM_RETAKE_GATE ->
                    "Wave-2 B3 active: retake gate now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRM_MISMATCH_GATE ->
                    "Wave-2 B4 active: mismatch gate now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRM_CONFLICT_GATE ->
                    "Wave-2 B5 active: conflict gate now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE ->
                    "Wave-2 B6 active: not-unique / blocked confirming gate now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CONFIRMING_VALIDATION_SUMMARY ->
                    "Transitional alias only: telemetry/reporting should treat this as CONFIRM_STATUS_SUMMARY."


                ReplyDemandCategoryV1.PENDING_CLARIFICATION ->
                    "Wave-1 C7 active: bounded clarification turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PENDING_CELL_CONFIRM_AS_IS ->
                    "Wave-2 C1 active: bounded pending cell-confirm-as-is turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PENDING_CELL_CONFIRM_TO_DIGIT ->
                    "Wave-2 C2 active: bounded pending cell-confirm-to-digit turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PENDING_REGION_CONFIRM_AS_IS ->
                    "Wave-2 C3 active: bounded pending region-confirm-as-is turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PENDING_REGION_CONFIRM_TO_DIGITS ->
                    "Wave-2 C4 active: bounded pending region-confirm-to-digits turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PENDING_DIGIT_PROVIDE ->
                    "Wave-2 C5 active: bounded pending digit-provide turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PENDING_INTERPRETATION_CONFIRM ->
                    "Wave-2 C6 active: bounded pending interpretation-confirm turn now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_VALIDATION_ANSWER ->
                    "Wave-1 D1 active: user-owned validation/inspection answer now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_CANDIDATE_ANSWER ->
                    "Wave-1 D4 active: user-owned candidate-state answer now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_OCR_TRUST_ANSWER ->
                    "Wave-3 D2 active: OCR/trust answer now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_CONTENTS_ANSWER ->
                    "Wave-3 D3 active: grid-contents answer now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_CHANGELOG_ANSWER ->
                    "Wave-3 D5 active: grid-changelog answer now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_EDIT_EXECUTION ->
                    "Wave-3 E1 active: direct grid-edit execution now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_CLEAR_EXECUTION ->
                    "Wave-3 E2 active: clear/erase execution now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_SWAP_EXECUTION ->
                    "Wave-3 E3 active: swap execution now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_BATCH_EDIT_EXECUTION ->
                    "Wave-3 E4 active: batch-edit execution now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_UNDO_REDO_EXECUTION ->
                    "Wave-3 E5 active: undo/redo execution now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GRID_LOCK_GIVENS_EXECUTION ->
                    "Wave-3 E6 active: lock-givens execution now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_STAGE_ELABORATION ->
                    "Wave-4 G1 active: in-lane stage elaboration now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_STAGE_REPEAT ->
                    "Wave-4 G2 active: in-lane stage repeat now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_STAGE_REPHRASE ->
                    "Wave-4 G3 active: in-lane stage rephrase now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_GO_BACKWARD ->
                    "Wave-4 G4 active: in-lane solving backward navigation now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_STEP_REVEAL ->
                    "Wave-4 G5 active: in-lane step reveal now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_ROUTE_CONTROL ->
                    "Wave-4 G6 active: in-lane route control now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE ->
                    "Series-H P8 active: proof-challenge detour now uses storyteller-parity projected prompt contract plus required personalization support so the same Sudo voice carries through setup, confrontation, resolution, and proof-challenge detours."

                ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY ->
                    "Wave-4 H2 active: target-cell detour query now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY ->
                    "Wave-4 H3 active: neighbor-cell detour query now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.DETOUR_REASONING_CHECK ->
                    "Wave-4 H4 active: user reasoning-check detour now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE ->
                    "Wave-4 H5 active: alternative-technique detour now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH ->
                    "Wave-4 H6 active: local-move-search detour now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON ->
                    "Wave-4 H7 active: route-comparison detour now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.PREFERENCE_CHANGE ->
                    "Wave-5 I1 active: preference-change turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.MODE_CHANGE ->
                    "Wave-5 I2 active: mode-change turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.ASSISTANT_PAUSE_RESUME ->
                    "Wave-5 I3 active: assistant pause/resume turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.VALIDATE_ONLY_OR_SOLVE_ONLY ->
                    "Wave-5 I4 active: validate-only / solve-only turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.FOCUS_REDIRECT ->
                    "Wave-5 I5 active: focus-redirect turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.HINT_POLICY_CHANGE ->
                    "Wave-5 I6 active: hint-policy-change turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.META_STATE_ANSWER ->
                    "Wave-5 J1 active: meta-state answer turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.CAPABILITY_ANSWER ->
                    "Wave-5 J2 active: capability-answer turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.GLOSSARY_ANSWER ->
                    "Wave-5 J3 active: glossary-answer turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.UI_HELP_ANSWER ->
                    "Wave-5 J4 active: UI-help turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.COORDINATE_HELP_ANSWER ->
                    "Wave-5 J5 active: coordinate-help turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.FREE_TALK_NON_GRID ->
                    "Wave-5 K1 active: true non-grid free talk now uses a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SMALL_TALK_BRIDGE ->
                    "Wave-5 K2 active: small-talk bridge turns now use a dedicated projected prompt contract."

                ReplyDemandCategoryV1.SOLVING_SETUP ->
                    if (ENABLE_SOLVING_SETUP_PACKET_V1) {
                        "Solving setup rollout active: canonical setup packet path is enabled."
                    } else {
                        "Solving setup rollout disabled: setup remains on legacy payload path."
                    }

                ReplyDemandCategoryV1.SOLVING_CONFRONTATION ->
                    if (ENABLE_SOLVING_CONFRONTATION_PACKET_V1) {
                        "Solving confrontation rollout active: canonical confrontation packet path is enabled with bounded proof-row shaping."
                    } else {
                        "Solving confrontation now uses slim prompt composition and projected proof-scope body shaping."
                    }

                ReplyDemandCategoryV1.SOLVING_RESOLUTION ->
                    if (ENABLE_SOLVING_RESOLUTION_PACKET_V1) {
                        "Solving resolution rollout active: canonical resolution packet path is enabled with bounded compact recap shaping."
                    } else {
                        "Solving resolution now uses slim prompt composition and projected resolution-body shaping."
                    }

                ReplyDemandCategoryV1.REPAIR_CONTRADICTION ->
                    "Repair contradiction now uses slim prompt composition and projected repair-body shaping."

                else ->
                    "Planner active; payload shaping still gated until later Phase 7 rollouts."
            }

        val notes =
            if (canonicalStoryMini != null &&
                (demand.category == ReplyDemandCategoryV1.SOLVING_SETUP ||
                        demand.category == ReplyDemandCategoryV1.SOLVING_CONFRONTATION ||
                        demand.category == ReplyDemandCategoryV1.SOLVING_RESOLUTION)
            ) {
                "$baseNotes Canonical story context: $canonicalStoryMini."
            } else {
                baseNotes
            }

        val plan = ReplyAssemblyPlanV1(
            demand = demand,
            contract = contract,
            selectedPromptModules = selectedPromptModules,
            selectedChannels = selectedChannels,
            rolloutMode = rolloutMode,
            notes = notes
        )

        return PlanResultV1(
            plan = plan,
            projectedChannels = projected
        )
    }
}