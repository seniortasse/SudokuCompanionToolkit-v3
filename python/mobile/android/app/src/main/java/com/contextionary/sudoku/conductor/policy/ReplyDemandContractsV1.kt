package com.contextionary.sudoku.conductor.policy



import com.contextionary.sudoku.conductor.GridPhase
import com.contextionary.sudoku.conductor.Pending
import com.contextionary.sudoku.conductor.TurnOwnerV1
import com.contextionary.sudoku.conductor.UserAgendaItem
import com.contextionary.sudoku.conductor.UserAgendaStatusV1

/**
 * Phase 3 — dedicated registry for reply demand contracts.
 *
 * This centralizes the supply-vs-demand table outside SudoConductor so later
 * phases can:
 * - add projector-based channel shaping
 * - add demand-specific prompt composition
 * - add budget validation
 * without bloating the conductor.
 *
 * Important:
 * Phase 3 is still behavior-neutral. These contracts are emitted to telemetry
 * but do not yet shape Tick2 payload assembly.
 */
object ReplyDemandContractsV1 {

    fun forDemand(
        demand: ReplyDemandResolutionV1
    ): ReplyDemandContractV1 {
        return when (demand.category) {
            ReplyDemandCategoryV1.ONBOARDING_OPENING -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.ONBOARDING_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.ONBOARDING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),


                budget = ReplyBudgetV1(
                    softCharBudget = 4400,
                    softTokenBudget = 1100
                ),
                notes = "Opening turn should stay workflow-introduction only and now uses slim prompt composition."



            )

            ReplyDemandCategoryV1.CONFIRM_STATUS_SUMMARY -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_STATUS_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-1 B1. Status summary only: summarize validation / solvability / seal state without turning the reply into a transactional gate."
            )

            ReplyDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_EXACT_MATCH_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-1 B2. Exact-match gate only: ask for or acknowledge exact screen-vs-puzzle confirmation without drifting into strategy chat."
            )

            ReplyDemandCategoryV1.CONFIRM_FINALIZE_GATE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_FINALIZE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5000,
                    softTokenBudget = 1250
                ),
                notes = "Wave-1 B7. Finalize handoff only: close confirming cleanly and ask the precise start-solving readiness CTA."
            )

            ReplyDemandCategoryV1.CONFIRM_RETAKE_GATE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_RETAKE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-2 B3. Retake gate only: ask whether to keep the scan or retake, and ground the ask in confirming truth."
            )

            ReplyDemandCategoryV1.CONFIRM_MISMATCH_GATE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_MISMATCH_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL

                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5400,
                    softTokenBudget = 1350
                ),
                notes = "Wave-2 B4. Mismatch gate only: confirm or repair screen-versus-puzzle mismatches using bounded validation truth."
            )

            ReplyDemandCategoryV1.CONFIRM_CONFLICT_GATE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_CONFLICT_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL

                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5400,
                    softTokenBudget = 1350
                ),
                notes = "Wave-2 B5. Conflict gate only: confirm or repair a conflict-bearing cell or region using conflict-grounded validation truth."
            )

            ReplyDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_NOT_UNIQUE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-2 B6. Not-unique / structurally blocked gate only: communicate non-uniqueness or non-solvability cleanly and guide the next confirming choice."
            )

            ReplyDemandCategoryV1.CONFIRMING_VALIDATION_SUMMARY -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.CONFIRM_STATUS_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Compatibility alias only: this legacy family now delegates to the modern CONFIRM_STATUS_SUMMARY contract surface."
            )

            ReplyDemandCategoryV1.PENDING_CLARIFICATION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.CLARIFICATION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3200,
                    softTokenBudget = 800
                ),
                notes = "Wave-1 C7. Clarification-only turn: ask exactly one bounded missing-target question and do not drift into strategy or unrelated CTA."
            )

            ReplyDemandCategoryV1.PENDING_CELL_CONFIRM_AS_IS -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.PENDING_CELL_CONFIRM_AS_IS_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3600,
                    softTokenBudget = 900
                ),
                notes = "Wave-2 C1. Bounded pending contract: confirm that the prompted cell is already correct."
            )

            ReplyDemandCategoryV1.PENDING_CELL_CONFIRM_TO_DIGIT -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.PENDING_CELL_CONFIRM_TO_DIGIT_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3600,
                    softTokenBudget = 900
                ),
                notes = "Wave-2 C2. Bounded pending contract: confirm the corrected digit for the prompted cell."
            )

            ReplyDemandCategoryV1.PENDING_REGION_CONFIRM_AS_IS -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.PENDING_REGION_CONFIRM_AS_IS_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3800,
                    softTokenBudget = 950
                ),
                notes = "Wave-2 C3. Bounded pending contract: confirm that the prompted row / column / box is already correct."
            )

            ReplyDemandCategoryV1.PENDING_REGION_CONFIRM_TO_DIGITS -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.PENDING_REGION_CONFIRM_TO_DIGITS_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4000,
                    softTokenBudget = 1000
                ),
                notes = "Wave-2 C4. Bounded pending contract: confirm the corrected digits for the prompted row / column / box."
            )

            ReplyDemandCategoryV1.PENDING_DIGIT_PROVIDE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.PENDING_DIGIT_PROVIDE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3400,
                    softTokenBudget = 850
                ),
                notes = "Wave-2 C5. Bounded pending contract: consume or request the specific digit needed by the current pending job."
            )

            ReplyDemandCategoryV1.PENDING_INTERPRETATION_CONFIRM -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.PENDING_GATE_RULES,
                    PromptModuleV1.PENDING_INTERPRETATION_CONFIRM_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3600,
                    softTokenBudget = 900
                ),
                notes = "Wave-2 C6. Bounded pending contract: confirm the app's current interpretation of the targeted cell / region / scan reading."
            )

            ReplyDemandCategoryV1.GRID_VALIDATION_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_VALIDATION_ANSWER_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-1 D1. Validation-answer turn: answer user-owned grid validity / conflicts / mismatches / unresolved-state questions directly."
            )

            ReplyDemandCategoryV1.GRID_CANDIDATE_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_CANDIDATE_ANSWER_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-1 D4. Candidate-answer turn: answer user-owned candidate-state questions directly without falling into broad grid free talk."
            )

            ReplyDemandCategoryV1.GRID_OCR_TRUST_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_OCR_TRUST_ANSWER_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4800,
                    softTokenBudget = 1200
                ),
                notes = "Wave-3 D2. OCR/trust-answer turn: answer user-owned scan-confidence and certainty questions directly."
            )

            ReplyDemandCategoryV1.GRID_CONTENTS_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_CONTENTS_ANSWER_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5200,
                    softTokenBudget = 1300
                ),
                notes = "Wave-3 D3. Grid-contents answer turn: answer user-owned row/column/box/value/missing-digits questions directly."
            )

            ReplyDemandCategoryV1.GRID_CHANGELOG_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_CHANGELOG_ANSWER_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Wave-3 D5. Changelog-answer turn: answer user-owned recent-change and mutation-result questions directly."
            )

            ReplyDemandCategoryV1.GRID_EDIT_EXECUTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_EDIT_EXECUTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4400,
                    softTokenBudget = 1100
                ),
                notes = "Wave-3 E1. Grid-edit execution turn: acknowledge or explain a direct cell edit mutation and its immediate result."
            )

            ReplyDemandCategoryV1.GRID_CLEAR_EXECUTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_CLEAR_EXECUTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Wave-3 E2. Grid-clear execution turn: acknowledge or explain a clear/erase mutation and its immediate result."
            )

            ReplyDemandCategoryV1.GRID_SWAP_EXECUTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_SWAP_EXECUTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4400,
                    softTokenBudget = 1100
                ),
                notes = "Wave-3 E3. Grid-swap execution turn: acknowledge or explain a swap mutation and its immediate result."
            )

            ReplyDemandCategoryV1.GRID_BATCH_EDIT_EXECUTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_BATCH_EDIT_EXECUTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5000,
                    softTokenBudget = 1250
                ),
                notes = "Wave-3 E4. Grid-batch-edit execution turn: acknowledge or explain a multi-edit mutation batch and its immediate result."
            )

            ReplyDemandCategoryV1.GRID_UNDO_REDO_EXECUTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_UNDO_REDO_EXECUTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Wave-3 E5. Undo/redo execution turn: acknowledge or explain an undo/redo mutation and its immediate result."
            )

            ReplyDemandCategoryV1.GRID_LOCK_GIVENS_EXECUTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.GRID_LOCK_GIVENS_EXECUTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Wave-3 E6. Lock-givens execution turn: acknowledge or explain locking scanned givens and its immediate result."
            )

            ReplyDemandCategoryV1.SOLVING_STAGE_ELABORATION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.SOLVING_STAGE_ELABORATION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5600,
                    softTokenBudget = 1400
                ),
                notes = "Wave-4 G1. In-lane solving elaboration turn: deepen the current solving stage without leaving the solving road."
            )

            ReplyDemandCategoryV1.SOLVING_STAGE_REPEAT -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.SOLVING_STAGE_REPEAT_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4800,
                    softTokenBudget = 1200
                ),
                notes = "Wave-4 G2. In-lane solving repeat turn: replay the current stage cleanly without advancing."
            )

            ReplyDemandCategoryV1.SOLVING_STAGE_REPHRASE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.SOLVING_STAGE_REPHRASE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5000,
                    softTokenBudget = 1250
                ),
                notes = "Wave-4 G3. In-lane solving rephrase turn: restate the same current stage in different words without advancing."
            )

            ReplyDemandCategoryV1.SOLVING_GO_BACKWARD -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.SOLVING_GO_BACKWARD_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5400,
                    softTokenBudget = 1350
                ),
                notes = "Wave-4 G4. In-lane solving backward turn: go back to a prior stage or step within the solving road."
            )

            ReplyDemandCategoryV1.SOLVING_STEP_REVEAL -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.SOLVING_STEP_REVEAL_RULES,
                    PromptModuleV1.CTA_ENDING_RULES,
                    PromptModuleV1.COMMIT_TRUTH_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.OVERLAY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5000,
                    softTokenBudget = 1250
                ),
                notes = "Wave-4 G5. In-lane step-reveal turn: reveal the answer payload for the current solving step under spoiler and commit truth rules."
            )

            ReplyDemandCategoryV1.SOLVING_ROUTE_CONTROL -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.SOLVING_ROUTE_CONTROL_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4600,
                    softTokenBudget = 1150
                ),
                notes = "Wave-4 G6. In-lane route-control turn: honor continue/pause/return-to-route style requests without switching to off-road free talk."
            )

            ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE -> ReplyDemandContractV1(
                demandCategory = demand.category,

                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_PROOF_CHALLENGE_RULES,
                    PromptModuleV1.DETOUR_MOVE_PROOF_RULES,
                    PromptModuleV1.DETOUR_PROOF_MICRO_STAGE_RULES,
                    PromptModuleV1.DETOUR_PROOF_CLOSURE_CTA_RULES,
                    PromptModuleV1.DETOUR_PROOF_GEOMETRY_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),

                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_MOVE_PROOF_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),


                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 6200,
                    softTokenBudget = 1550
                ),
                notes = "Series-I P9. Proof-challenge detour now uses upstream normalization enrichment: story-owned typed move-proof packet + micro-stage shaping + authored closure + voice parity + richer normalized local story question / spotlight object / actor roles / proof motion / visible tension cues."
            )

            ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_TARGET_CELL_QUERY_RULES,
                    PromptModuleV1.DETOUR_MOVE_PROOF_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),

                requiredChannels = setOf(
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_MOVE_PROOF_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT
                ),


                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5600,
                    softTokenBudget = 1400
                ),
                notes = "Wave-4 H2 + Wave-1 F3. Target-cell-query detour turn now formally requires both shared DETOUR_CONTEXT and the typed wave-1 move-proof packet."
            )

            ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_NEIGHBOR_CELL_QUERY_RULES,
                    PromptModuleV1.DETOUR_LOCAL_GRID_INSPECTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),

                requiredChannels = setOf(
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_LOCAL_GRID_INSPECTION_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT
                ),


                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 5600,
                    softTokenBudget = 1400
                ),
                notes = "Wave-4 H3 + Wave-1 F3. Neighbor-cell-query detour turn now formally requires both shared DETOUR_CONTEXT and the typed wave-1 local-grid-inspection packet."
            )

            ReplyDemandCategoryV1.DETOUR_REASONING_CHECK -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_REASONING_CHECK_RULES,
                    PromptModuleV1.DETOUR_USER_PROPOSAL_VERDICT_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_USER_PROPOSAL_VERDICT_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 6200,
                    softTokenBudget = 1550
                ),
                notes = "Wave-4 H4 + Wave-1 F3. Reasoning-check detour turn now formally requires both shared DETOUR_CONTEXT and the typed wave-1 user-proposal-verdict packet."
            )

            ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_ALTERNATIVE_TECHNIQUE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),

                requiredChannels = setOf(
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_ALTERNATIVE_TECHNIQUE_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT
                ),


                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 6200,
                    softTokenBudget = 1550
                ),
                notes = "Track-1 Phase-1 / Plan-1B. Alternative-technique detour is now contract-level packet-centered: typed packet + detour narrative context are primary; shared DETOUR_CONTEXT is support only until the dedicated packet projector lands in Phase 2."
            )

            ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_LOCAL_MOVE_SEARCH_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),

                requiredChannels = setOf(
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_LOCAL_MOVE_SEARCH_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT
                ),


                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 6000,
                    softTokenBudget = 1500
                ),
                notes = "Track-1 Phase-1 / Plan-1B. Local-move-search detour is now contract-level packet-centered: typed packet + detour narrative context are primary; shared DETOUR_CONTEXT is support only until the dedicated packet projector lands in Phase 2."
            )

            ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GRID_TRUTH_RULES,
                    PromptModuleV1.DETOUR_ROUTE_COMPARISON_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),

                requiredChannels = setOf(
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.DETOUR_ROUTE_COMPARISON_PACKET,
                    ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT
                ),


                optionalChannels = setOf(
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 6200,
                    softTokenBudget = 1550
                ),
                notes = "Track-1 Phase-1 / Plan-1B. Route-comparison detour is now contract-level packet-centered: typed packet + detour narrative context are primary; shared DETOUR_CONTEXT is support only until the dedicated packet projector lands in Phase 2."
            )

            ReplyDemandCategoryV1.PREFERENCE_CHANGE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.PREFERENCE_CHANGE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.PREFERENCE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3200,
                    softTokenBudget = 800
                ),
                notes = "Wave-5 I1. Preference-change turn: acknowledge and apply conversational or teaching preference changes."
            )

            ReplyDemandCategoryV1.MODE_CHANGE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.MODE_CHANGE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.PREFERENCE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3200,
                    softTokenBudget = 800
                ),
                notes = "Wave-5 I2. Mode-change turn: explain or confirm a high-level mode switch."
            )

            ReplyDemandCategoryV1.ASSISTANT_PAUSE_RESUME -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.ASSISTANT_PAUSE_RESUME_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.PREFERENCE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 2600,
                    softTokenBudget = 650
                ),
                notes = "Wave-5 I3. Pause/resume turn: honor user requests to pause or resume the assistant."
            )

            ReplyDemandCategoryV1.VALIDATE_ONLY_OR_SOLVE_ONLY -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.VALIDATE_ONLY_OR_SOLVE_ONLY_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.PREFERENCE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3400,
                    softTokenBudget = 850
                ),
                notes = "Wave-5 I4. Validate-only / solve-only turn: acknowledge and set workflow constraints on the assistant."
            )

            ReplyDemandCategoryV1.FOCUS_REDIRECT -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.FOCUS_REDIRECT_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.PREFERENCE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3400,
                    softTokenBudget = 850
                ),
                notes = "Wave-5 I5. Focus-redirect turn: redirect assistant attention to a requested area or topic of focus."
            )

            ReplyDemandCategoryV1.HINT_POLICY_CHANGE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.HINT_POLICY_CHANGE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.PREFERENCE_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3400,
                    softTokenBudget = 850
                ),
                notes = "Wave-5 I6. Hint-policy change turn: acknowledge requested changes to hint strength or teaching policy."
            )

            ReplyDemandCategoryV1.META_STATE_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.META_STATE_ANSWER_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.META_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Wave-5 J1. Meta-state turn: answer what the assistant currently knows, is doing, or is tracking."
            )

            ReplyDemandCategoryV1.CAPABILITY_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.CAPABILITY_ANSWER_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.META_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3200,
                    softTokenBudget = 800
                ),
                notes = "Wave-5 J2. Capability-answer turn: answer what the assistant/app can or cannot do."
            )

            ReplyDemandCategoryV1.GLOSSARY_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.GLOSSARY_ANSWER_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.HELP_CONTEXT,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3800,
                    softTokenBudget = 950
                ),
                notes = "Wave-5 J3. Glossary-answer turn: define or explain terminology using glossary support."
            )

            ReplyDemandCategoryV1.UI_HELP_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.UI_HELP_ANSWER_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.HELP_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3600,
                    softTokenBudget = 900
                ),
                notes = "Wave-5 J4. UI-help turn: answer interface/legend/help questions."
            )

            ReplyDemandCategoryV1.COORDINATE_HELP_ANSWER -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.COORDINATE_HELP_ANSWER_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.HELP_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3600,
                    softTokenBudget = 900
                ),
                notes = "Wave-5 J5. Coordinate-help turn: explain coordinates, box indexing, and locating cells."
            )

            ReplyDemandCategoryV1.FREE_TALK_NON_GRID -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.FREE_TALK_NON_GRID_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.FREE_TALK_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT,
                    ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT,
                    ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT,
                    ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT,
                    ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT,
                    ReplySupplyChannelV1.DETOUR_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3600,
                    softTokenBudget = 900
                ),
                notes = "Wave-5 K1. True non-grid free-talk turn."
            )

            ReplyDemandCategoryV1.SMALL_TALK_BRIDGE -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.SMALL_TALK_BRIDGE_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.FREE_TALK_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                    ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL,
                    ReplySupplyChannelV1.GRID_MUTATION_CONTEXT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 2600,
                    softTokenBudget = 650
                ),
                notes = "Wave-5 K2. Lightweight small-talk bridge that preserves continuity without opening a broad detour."
            )

            ReplyDemandCategoryV1.SOLVING_SETUP -> {
                val setupDoctrinePromptModules =
                    when (demand.setupProfile) {
                        SetupDemandProfileV1.BASE_SINGLES_SETUP,
                        SetupDemandProfileV1.FULL_HOUSE_SETUP ->
                            setOf(PromptModuleV1.SETUP_LENS_FIRST_RULES)

                        SetupDemandProfileV1.SUBSETS_SETUP,
                        SetupDemandProfileV1.ADVANCED_PATTERN_SETUP ->
                            setOf(PromptModuleV1.SETUP_PATTERN_FIRST_RULES)

                        SetupDemandProfileV1.INTERSECTIONS_SETUP ->
                            setOf(PromptModuleV1.INTERSECTION_SETUP_RULES)

                        else -> emptySet()
                    }

                ReplyDemandContractV1(
                    demandCategory = demand.category,
                    requiredPromptModules = setOf(
                        PromptModuleV1.BASE_JSON_OUTPUT,
                        PromptModuleV1.BASE_PERSONA,
                        PromptModuleV1.NO_INVENTION_RULES,
                        PromptModuleV1.GRID_TRUTH_RULES,
                        PromptModuleV1.CTA_ENDING_RULES
                    ) + setupDoctrinePromptModules,
                    requiredChannels = setOf(
                        ReplySupplyChannelV1.TURN_HEADER_MINI,
                        ReplySupplyChannelV1.STYLE_MINI,
                        ReplySupplyChannelV1.CTA_CONTEXT,
                        ReplySupplyChannelV1.PERSONALIZATION_MINI,
                        ReplySupplyChannelV1.SETUP_REPLY_PACKET
                    ),
                    optionalChannels = setOf(
                        ReplySupplyChannelV1.CONTINUITY_SHORT
                    ),
                    forbiddenChannels = setOf(
                        ReplySupplyChannelV1.SETUP_STORY_SLICE,
                        ReplySupplyChannelV1.SETUP_STEP_SLICE,
                        ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                        ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                        ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                        ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                        ReplySupplyChannelV1.OVERLAY_MINI,
                        ReplySupplyChannelV1.REPAIR_CONTEXT,
                        ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                        ReplySupplyChannelV1.GLOSSARY_MINI,
                        ReplySupplyChannelV1.TECHNIQUE_CARD_MINI
                    ),
                    budget = ReplyBudgetV1(
                        softCharBudget = 9000,
                        softTokenBudget = 2250
                    ),
                    notes =
                        if (demand.setupProfile == SetupDemandProfileV1.FULL_HOUSE_SETUP) {
                            "Full House setup turn is packet-centered: lens-first prompt modules and SETUP_REPLY_PACKET dominate the surface; non-setup story/step slices stay forbidden."
                        } else {
                            "Setup turn is packet-centered: only required channels are selected so doctrine prompts and SETUP_REPLY_PACKET dominate the surface."
                        }
                )
            }

            ReplyDemandCategoryV1.SOLVING_CONFRONTATION -> {
                val confrontationDoctrinePromptModules =
                    when (demand.confrontationProofProfile) {
                        ConfrontationProofProfileV1.BASE_SINGLES_PROOF,
                        ConfrontationProofProfileV1.FULL_HOUSE_PROOF ->
                            setOf(PromptModuleV1.CONFRONTATION_LENS_FIRST_RULES)

                        ConfrontationProofProfileV1.SUBSETS_PROOF,
                        ConfrontationProofProfileV1.ADVANCED_PATTERN_PROOF ->
                            setOf(PromptModuleV1.CONFRONTATION_PATTERN_FIRST_RULES)

                        ConfrontationProofProfileV1.INTERSECTIONS_PROOF ->
                            setOf(
                                PromptModuleV1.CONFRONTATION_PATTERN_FIRST_RULES,
                                PromptModuleV1.INTERSECTION_CONFRONTATION_RULES
                            )

                        else -> emptySet()
                    }

                ReplyDemandContractV1(
                    demandCategory = demand.category,
                    requiredPromptModules = setOf(
                        PromptModuleV1.BASE_JSON_OUTPUT,
                        PromptModuleV1.BASE_PERSONA,
                        PromptModuleV1.NO_INVENTION_RULES,
                        PromptModuleV1.GRID_TRUTH_RULES,
                        PromptModuleV1.CTA_ENDING_RULES
                    ) + confrontationDoctrinePromptModules,
                    requiredChannels = setOf(
                        ReplySupplyChannelV1.TURN_HEADER_MINI,
                        ReplySupplyChannelV1.STYLE_MINI,
                        ReplySupplyChannelV1.CTA_CONTEXT,
                        ReplySupplyChannelV1.PERSONALIZATION_MINI,
                        ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET
                    ),
                    optionalChannels = setOf(
                        ReplySupplyChannelV1.CONTINUITY_SHORT
                    ),
                    forbiddenChannels = setOf(
                        ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                        ReplySupplyChannelV1.SETUP_STORY_SLICE,
                        ReplySupplyChannelV1.SETUP_STEP_SLICE,
                        ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                        ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                        ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                        ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                        ReplySupplyChannelV1.OVERLAY_MINI,
                        ReplySupplyChannelV1.REPAIR_CONTEXT,
                        ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                        ReplySupplyChannelV1.GLOSSARY_MINI,
                        ReplySupplyChannelV1.TECHNIQUE_CARD_MINI
                    ),
                    budget = ReplyBudgetV1(
                        softCharBudget = 14000,
                        softTokenBudget = 3500
                    ),
                    notes =
                        if (demand.confrontationProofProfile == ConfrontationProofProfileV1.FULL_HOUSE_PROOF) {
                            "Full House confrontation turn is packet-centered: lens-first confrontation prompts and CONFRONTATION_REPLY_PACKET dominate the surface; setup/resolution packet channels remain forbidden."
                        } else {
                            "Confrontation turn is packet-centered: only required channels are selected so confrontation doctrine prompts and CONFRONTATION_REPLY_PACKET dominate the surface."
                        }
                )
            }

            ReplyDemandCategoryV1.SOLVING_RESOLUTION -> {
                val resolutionDoctrinePromptModules =
                    when (demand.resolutionProfile) {
                        ResolutionProfileV1.BASE_SINGLES_RESOLUTION,
                        ResolutionProfileV1.FULL_HOUSE_RESOLUTION ->
                            setOf(PromptModuleV1.RESOLUTION_BASIC_RULES)

                        ResolutionProfileV1.SUBSETS_RESOLUTION,
                        ResolutionProfileV1.ADVANCED_PATTERN_RESOLUTION ->
                            setOf(PromptModuleV1.RESOLUTION_ADVANCED_RULES)

                        ResolutionProfileV1.INTERSECTIONS_RESOLUTION ->
                            setOf(
                                PromptModuleV1.RESOLUTION_ADVANCED_RULES,
                                PromptModuleV1.INTERSECTION_RESOLUTION_RULES
                            )

                        else -> emptySet()
                    }

                ReplyDemandContractV1(
                    demandCategory = demand.category,
                    requiredPromptModules = setOf(
                        PromptModuleV1.BASE_JSON_OUTPUT,
                        PromptModuleV1.BASE_PERSONA,
                        PromptModuleV1.GRID_TRUTH_RULES,
                        PromptModuleV1.SOLVING_RESOLUTION_RULES,
                        PromptModuleV1.COMMIT_TRUTH_RULES,
                        PromptModuleV1.CTA_ENDING_RULES
                    ) + resolutionDoctrinePromptModules,
                    requiredChannels = setOf(
                        ReplySupplyChannelV1.TURN_HEADER_MINI,
                        ReplySupplyChannelV1.STYLE_MINI,
                        ReplySupplyChannelV1.CTA_CONTEXT,
                        ReplySupplyChannelV1.PERSONALIZATION_MINI,
                        ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET
                    ),
                    optionalChannels = setOf(
                        ReplySupplyChannelV1.CONTINUITY_SHORT
                    ),
                    forbiddenChannels = setOf(
                        ReplySupplyChannelV1.SETUP_REPLY_PACKET,
                        ReplySupplyChannelV1.SETUP_STORY_SLICE,
                        ReplySupplyChannelV1.SETUP_STEP_SLICE,
                        ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET,
                        ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                        ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                        ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                        ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                        ReplySupplyChannelV1.OVERLAY_MINI,
                        ReplySupplyChannelV1.REPAIR_CONTEXT,
                        ReplySupplyChannelV1.CONFIRMING_CONTEXT,
                        ReplySupplyChannelV1.GLOSSARY_MINI,
                        ReplySupplyChannelV1.TECHNIQUE_CARD_MINI
                    ),
                    budget = ReplyBudgetV1(
                        softCharBudget = 9000,
                        softTokenBudget = 2250
                    ),
                    notes =
                        if (demand.resolutionProfile == ResolutionProfileV1.FULL_HOUSE_RESOLUTION) {
                            "Full House resolution turn is packet-centered: RESOLUTION_REPLY_PACKET owns the surface, with compact commit truth, recap, lesson-style technique contribution, and next-step CTA."
                        } else {
                            "Resolution turn is packet-centered: commit truth + compact recap + technique contribution + final forcing + next-step CTA."
                        }
                )
            }

            ReplyDemandCategoryV1.REPAIR_CONTRADICTION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.REPAIR_RULES,
                    PromptModuleV1.NO_CONTRADICTION_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CTA_CONTEXT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI,
                    ReplySupplyChannelV1.SOLVABILITY_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 3250,
                    softTokenBudget = 813
                ),
                notes = "Repair reply should not drag solving payload into Tick2 and now uses projected repair-body shaping."
            )

            ReplyDemandCategoryV1.FREE_TALK_IN_GRID_SESSION -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.META_STATE_ANSWER_RULES,
                    PromptModuleV1.CTA_ENDING_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI,
                    ReplySupplyChannelV1.META_CONTEXT
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.DECISION_SUMMARY_MINI,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.HANDOVER_NOTE_MINI
                ),
                forbiddenChannels = setOf(
                    ReplySupplyChannelV1.SETUP_STORY_SLICE,
                    ReplySupplyChannelV1.SETUP_STEP_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE,
                    ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STORY_SLICE,
                    ReplySupplyChannelV1.RESOLUTION_STEP_SLICE,
                    ReplySupplyChannelV1.OVERLAY_MINI,
                    ReplySupplyChannelV1.GLOSSARY_MINI,
                    ReplySupplyChannelV1.TECHNIQUE_CARD_MINI,
                    ReplySupplyChannelV1.REPAIR_CONTEXT
                ),
                budget = ReplyBudgetV1(
                    softCharBudget = 4200,
                    softTokenBudget = 1050
                ),
                notes = "Compatibility alias only: this legacy family now delegates to the modern META_STATE_ANSWER contract surface."
            )



            ReplyDemandCategoryV1.RECOVERY_REPLY -> ReplyDemandContractV1(
                demandCategory = demand.category,
                requiredPromptModules = setOf(
                    PromptModuleV1.BASE_JSON_OUTPUT,
                    PromptModuleV1.BASE_PERSONA,
                    PromptModuleV1.NO_CONTRADICTION_RULES
                ),
                requiredChannels = setOf(
                    ReplySupplyChannelV1.TURN_HEADER_MINI,
                    ReplySupplyChannelV1.STYLE_MINI
                ),
                optionalChannels = setOf(
                    ReplySupplyChannelV1.CONTINUITY_SHORT,
                    ReplySupplyChannelV1.PERSONALIZATION_MINI,
                    ReplySupplyChannelV1.CTA_CONTEXT
                ),
                forbiddenChannels = emptySet(),
                budget = ReplyBudgetV1(
                    softCharBudget = 5000,
                    softTokenBudget = 1250
                ),
                notes = "Recovery category is intentionally permissive until later phases."
            )
        }
    }

    fun deriveCtaRouteMomentV1(
        ownerKind: TurnOwnerV1,
        phase: GridPhase,
        pending: Pending?,
        replyDemand: ReplyDemandResolutionV1,
        currentUserAgendaHead: UserAgendaItem? = null,
        facts: List<FactBundleV1> = emptyList()
    ): CtaRouteMomentV1 {

        val hasLocalMismatchDetour =
            facts.any {
                it.type == FactBundleV1.Type.OTHER &&
                        (
                                it.payload.optString("kind") == "grid_mismatch_report_detour" ||
                                        it.payload.optString("kind") == "grid_mismatch_report"
                                )
            }

        return when (ownerKind) {
            TurnOwnerV1.APP_OWNER -> {
                when {
                    pending is Pending.AfterResolution ->
                        CtaRouteMomentV1.SOLVING_POST_COMMIT

                    replyDemand.category == ReplyDemandCategoryV1.SOLVING_SETUP ->
                        CtaRouteMomentV1.SOLVING_SETUP

                    replyDemand.category == ReplyDemandCategoryV1.SOLVING_CONFRONTATION ->
                        CtaRouteMomentV1.SOLVING_CONFRONTATION

                    replyDemand.category == ReplyDemandCategoryV1.SOLVING_RESOLUTION ->
                        CtaRouteMomentV1.SOLVING_RESOLUTION

                    phase == GridPhase.CONFIRMING ->
                        CtaRouteMomentV1.CONFIRMING

                    phase == GridPhase.SEALING ->
                        CtaRouteMomentV1.SEALING

                    else -> CtaRouteMomentV1.UNKNOWN
                }
            }

            TurnOwnerV1.USER_OWNER -> {
                when {
                    pending is Pending.ConfirmRetake || pending is Pending.ConfirmValidate ->
                        CtaRouteMomentV1.USER_ROUTE_MUTATION_DECISION

                    pending is Pending.AskClarification || pending is Pending.ConfirmInterpretation ->
                        CtaRouteMomentV1.USER_DETOUR_PARTIAL

                    hasLocalMismatchDetour ->
                        CtaRouteMomentV1.USER_LOCAL_REPAIR_DECISION

                    pending is Pending.UserAgendaBridge ->
                        CtaRouteMomentV1.USER_DETOUR_COMPLETE

                    else -> CtaRouteMomentV1.UNKNOWN
                }
            }

            TurnOwnerV1.NONE -> {
                when (phase) {
                    GridPhase.CONFIRMING -> CtaRouteMomentV1.CONFIRMING
                    GridPhase.SEALING -> CtaRouteMomentV1.SEALING
                    else -> CtaRouteMomentV1.UNKNOWN
                }
            }
        }
    }

    fun policyForRouteMomentV1(
        routeMoment: CtaRouteMomentV1
    ): CtaPolicyV1 {
        return when (routeMoment) {
            CtaRouteMomentV1.SOLVING_SETUP ->
                CtaPolicyV1(
                    family = CtaFamilyV1.APP_SETUP_DISCOVERY,
                    expectedResponseType = CtaResponseTypeV1.NAME_DIGIT,
                    askMode = CtaAskModeV1.DIRECT_QUESTION,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = true
                )

            CtaRouteMomentV1.SOLVING_CONFRONTATION ->
                CtaPolicyV1(
                    family = CtaFamilyV1.APP_CONFRONTATION_PROOF_STEP,
                    expectedResponseType = CtaResponseTypeV1.CONFIRM_REASONING,
                    askMode = CtaAskModeV1.GUIDED_PROMPT,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = true
                )

            CtaRouteMomentV1.SOLVING_RESOLUTION ->
                CtaPolicyV1(
                    family = CtaFamilyV1.APP_RESOLUTION_COMMIT,
                    expectedResponseType = CtaResponseTypeV1.PERMISSION_TO_APPLY,
                    askMode = CtaAskModeV1.PERMISSION_ASK,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = true
                )

            CtaRouteMomentV1.SOLVING_POST_COMMIT ->
                CtaPolicyV1(
                    family = CtaFamilyV1.APP_POST_COMMIT_CONTINUE,
                    expectedResponseType = CtaResponseTypeV1.CONTINUE_OR_PAUSE,
                    askMode = CtaAskModeV1.BINARY_CHOICE,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = false,
                    mustReferenceFocusScope = false
                )

            CtaRouteMomentV1.USER_DETOUR_COMPLETE ->
                CtaPolicyV1(
                    family = CtaFamilyV1.USER_DETOUR_FOLLOWUP_OR_RETURN,
                    expectedResponseType = CtaResponseTypeV1.CHOOSE_ONE_OF_TWO,
                    askMode = CtaAskModeV1.BINARY_CHOICE,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = true,
                    mustOfferReturnChoice = true,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = false
                )

            CtaRouteMomentV1.USER_DETOUR_PARTIAL ->
                CtaPolicyV1(
                    family = CtaFamilyV1.USER_DETOUR_FOLLOWUP_ONLY,
                    expectedResponseType = CtaResponseTypeV1.CLARIFY_SCOPE,
                    askMode = CtaAskModeV1.CLARIFYING_QUESTION,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = true
                )

            CtaRouteMomentV1.USER_ROUTE_MUTATION_DECISION ->
                CtaPolicyV1(
                    family = CtaFamilyV1.USER_ROUTE_CONTROL_CONFIRM,
                    expectedResponseType = CtaResponseTypeV1.YES_NO,
                    askMode = CtaAskModeV1.BINARY_CHOICE,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = false
                )

            CtaRouteMomentV1.USER_LOCAL_REPAIR_DECISION ->
                CtaPolicyV1(
                    family = CtaFamilyV1.USER_LOCAL_REPAIR_CONFIRM,
                    expectedResponseType = CtaResponseTypeV1.CHOOSE_ONE_OF_TWO,
                    askMode = CtaAskModeV1.BINARY_CHOICE,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = true,
                    mustNotAdvanceStage = true,
                    mustReferenceFocusScope = true
                )

            else ->
                CtaPolicyV1(
                    family = CtaFamilyV1.UNKNOWN,
                    expectedResponseType = CtaResponseTypeV1.UNKNOWN,
                    askMode = CtaAskModeV1.UNKNOWN,
                    allowInternalJargon = false,
                    mustOfferFollowUpChoice = false,
                    mustOfferReturnChoice = false,
                    mustNotAdvanceStage = false,
                    mustReferenceFocusScope = false
                )
        }
    }

    fun buildCtaContractV1(
        ownerKind: TurnOwnerV1,
        routeMoment: CtaRouteMomentV1,
        focusCellRef: String? = null,
        focusHouseRef: String? = null,
        focusDigit: Int? = null,
        techniqueName: String? = null,
        bannedPhrases: List<String> = emptyList()
    ): CtaContractV1 {
        val policy = policyForRouteMomentV1(routeMoment)

        val closureIntent: String?
        val bridgeIntent: String?
        val askIntent: String?

        when (policy.family) {
            CtaFamilyV1.APP_SETUP_DISCOVERY -> {
                closureIntent = "close_setup"
                bridgeIntent = "focus_discovery_scope"
                askIntent = "identify_forced_digit_or_cell"
            }

            CtaFamilyV1.APP_CONFRONTATION_PROOF_STEP -> {
                closureIntent = "close_current_proof_point"
                bridgeIntent = "move_to_next_proof_check"
                askIntent = "request_next_inference"
            }

            CtaFamilyV1.APP_RESOLUTION_COMMIT -> {
                closureIntent = "state_proved_conclusion"
                bridgeIntent = "connect_proof_to_action"
                askIntent = "ask_to_apply_or_confirm_placement"
            }

            CtaFamilyV1.APP_POST_COMMIT_CONTINUE -> {
                closureIntent = "close_committed_step"
                bridgeIntent = "offer_next_step"
                askIntent = "continue_or_pause"
            }

            CtaFamilyV1.USER_DETOUR_FOLLOWUP_OR_RETURN -> {
                closureIntent = "close_detour_answer"
                bridgeIntent = "return_to_paused_move_or_allow_followup"
                askIntent = "choose_return_or_followup"
            }

            CtaFamilyV1.USER_DETOUR_FOLLOWUP_ONLY -> {
                closureIntent = "mark_detour_as_partial"
                bridgeIntent = "stay_on_same_detour"
                askIntent = "clarify_scope"
            }

            CtaFamilyV1.USER_ROUTE_CONTROL_CONFIRM -> {
                closureIntent = "confirm_high_impact_route_change"
                bridgeIntent = "hold_route_until_confirmed"
                askIntent = "confirm_route_mutation"
            }

            CtaFamilyV1.USER_LOCAL_REPAIR_CONFIRM -> {
                closureIntent = "acknowledge_local_discrepancy"
                bridgeIntent = "decide_local_fix_vs_broader_recheck"
                askIntent = "choose_local_or_global_repair"
            }

            CtaFamilyV1.UNKNOWN -> {
                closureIntent = null
                bridgeIntent = null
                askIntent = null
            }
        }

        return CtaContractV1(
            family = policy.family,
            ownerKind = ownerKind.name,
            routeMoment = routeMoment,
            expectedResponseType = policy.expectedResponseType,
            askMode = policy.askMode,
            closureIntent = closureIntent,
            bridgeIntent = bridgeIntent,
            askIntent = askIntent,
            focusCellRef = focusCellRef,
            focusHouseRef = focusHouseRef,
            focusDigit = focusDigit,
            techniqueName = techniqueName,
            allowFollowUp = when (policy.family) {
                CtaFamilyV1.USER_ROUTE_CONTROL_CONFIRM -> false
                else -> true
            },
            allowReturnToRoute = policy.mustOfferReturnChoice,
            allowRouteMutation = policy.family == CtaFamilyV1.USER_ROUTE_CONTROL_CONFIRM ||
                    policy.family == CtaFamilyV1.USER_LOCAL_REPAIR_CONFIRM,
            bannedPhrases = bannedPhrases,
            toneStyle = when (policy.family) {
                CtaFamilyV1.APP_RESOLUTION_COMMIT,
                CtaFamilyV1.USER_ROUTE_CONTROL_CONFIRM -> CtaToneStyleV1.DECISIVE
                CtaFamilyV1.APP_CONFRONTATION_PROOF_STEP -> CtaToneStyleV1.EXPLORATORY
                else -> CtaToneStyleV1.WARM_GUIDE
            },
            policy = policy
        )
    }
}