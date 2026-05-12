package com.contextionary.sudoku.conductor.solving

import com.contextionary.sudoku.conductor.policy.DetourAnswerBoundaryV1
import com.contextionary.sudoku.conductor.policy.DetourDemandCategoryV2
import com.contextionary.sudoku.conductor.policy.DetourHandbackPolicyV1
import com.contextionary.sudoku.conductor.policy.DetourHandoverModeV1
import com.contextionary.sudoku.conductor.policy.DetourNarrativeArchetypeV1
import com.contextionary.sudoku.conductor.policy.DetourOverlayPolicyV1
import org.json.JSONArray
import org.json.JSONObject

/**
 * DetourNarrativeModelsV1
 *
 * Permanent-design native narrative models for user-owned detours.
 *
 * Why this file exists:
 * - Main-road solving already has a native narrative model layer.
 * - Detours should not be architecturally second-class or rely on a reply-only
 *   "bridge atom" channel as their true home.
 * - This file is the native detour sibling to NarrativeAtomModelsV1.kt.
 *
 * Important scope of Phase F6R:
 * - model layer only
 * - no packet rewiring yet
 * - no builders yet
 * - no projector ownership yet
 *
 * Builders from canonical normalized detour truth are added in later phases.
 */


/* -------------------------------------------------------------------------
 * Shared helpers
 * ------------------------------------------------------------------------- */

private fun jsonArrayOfStringsV1(values: List<String>): JSONArray =
    JSONArray().apply { values.forEach { put(it) } }

private fun jsonArrayOfIntsV1(values: List<Int>): JSONArray =
    JSONArray().apply { values.forEach { put(it) } }

private fun jsonArrayOfEnumNamesV1(values: List<DetourAnswerBoundaryV1>): JSONArray =
    JSONArray().apply { values.forEach { put(it.name) } }


/* -------------------------------------------------------------------------
 * Native detour visual contract
 * ------------------------------------------------------------------------- */

data class DetourVisualCellV1(
    val cellRef: String,
    val role: String,
    val digit: Int? = null,
    val emphasis: String? = null,
    val candidateDigits: List<Int> = emptyList(),
    val meta: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("cell_ref", cellRef)
        put("role", role)
        put("digit", digit ?: JSONObject.NULL)
        put("emphasis", emphasis ?: JSONObject.NULL)
        put("candidate_digits", jsonArrayOfIntsV1(candidateDigits))
        put("meta", meta)
    }
}

data class DetourVisualHouseV1(
    val houseRef: String,
    val role: String,
    val emphasis: String? = null,
    val meta: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("house_ref", houseRef)
        put("role", role)
        put("emphasis", emphasis ?: JSONObject.NULL)
        put("meta", meta)
    }
}

data class DetourVisualCandidateMarkV1(
    val cellRef: String,
    val digit: Int,
    val role: String,
    val emphasis: String? = null,
    val meta: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("cell_ref", cellRef)
        put("digit", digit)
        put("role", role)
        put("emphasis", emphasis ?: JSONObject.NULL)
        put("meta", meta)
    }
}

data class DetourVisualContractV1(
    val storyKind: String? = null,
    val focusCell: String? = null,
    val primaryCells: List<DetourVisualCellV1> = emptyList(),
    val secondaryCells: List<DetourVisualCellV1> = emptyList(),
    val houses: List<DetourVisualHouseV1> = emptyList(),
    val candidateMarks: List<DetourVisualCandidateMarkV1> = emptyList(),
    val deemphasizeCells: List<String> = emptyList(),
    val availableLayers: List<String> = emptyList(),
    val defaultAppliedLayers: List<String> = emptyList(),
    val sceneNotes: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("story_kind", storyKind ?: JSONObject.NULL)
        put("focus_cell", focusCell ?: JSONObject.NULL)
        put("primary_cells", JSONArray().apply { primaryCells.forEach { put(it.toJson()) } })
        put("secondary_cells", JSONArray().apply { secondaryCells.forEach { put(it.toJson()) } })
        put("houses", JSONArray().apply { houses.forEach { put(it.toJson()) } })
        put("candidate_marks", JSONArray().apply { candidateMarks.forEach { put(it.toJson()) } })
        put("deemphasize_cells", jsonArrayOfStringsV1(deemphasizeCells))
        put("available_layers", jsonArrayOfStringsV1(availableLayers))
        put("default_applied_layers", jsonArrayOfStringsV1(defaultAppliedLayers))
        put("scene_notes", sceneNotes)
    }
}


/* -------------------------------------------------------------------------
 * Base atom contract
 * ------------------------------------------------------------------------- */

sealed interface DetourNarrativeAtomV1 {
    val atomKind: String
    val demandCategory: DetourDemandCategoryV2
    val archetype: DetourNarrativeArchetypeV1

    val focusScope: String?
    val focusCells: List<String>
    val focusHouses: List<String>

    val answerBoundary: List<DetourAnswerBoundaryV1>
    val handoverMode: DetourHandoverModeV1

    fun toJson(): JSONObject
}


/* -------------------------------------------------------------------------
 * Wave 1 atom 1 — local proof spotlight
 * ------------------------------------------------------------------------- */

data class LocalProofSpotlightAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.LOCAL_PROOF_SPOTLIGHT,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val decisiveFact: String? = null,
    val evidenceSummary: JSONArray = JSONArray(),
    val survivorSummary: JSONObject = JSONObject(),
    val allowedStageBoundary: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "LOCAL_PROOF_SPOTLIGHT"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)
        put("evidence_summary", evidenceSummary)
        put("survivor_summary", survivorSummary)
        put("allowed_stage_boundary", allowedStageBoundary ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}


data class ContradictionSpotlightAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val targetCell: String? = null,
    val askedDigit: Int? = null,
    val blockerCell: String? = null,
    val blockerHouse: String? = null,
    val decisiveFact: String? = null,
    val evidenceSummary: JSONArray = JSONArray(),

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "CONTRADICTION_SPOTLIGHT"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("target_cell", targetCell ?: JSONObject.NULL)
        put("asked_digit", askedDigit ?: JSONObject.NULL)
        put("blocker_cell", blockerCell ?: JSONObject.NULL)
        put("blocker_house", blockerHouse ?: JSONObject.NULL)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)
        put("evidence_summary", evidenceSummary)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class LocalPermissibilityScanAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val targetCell: String? = null,
    val askedDigit: Int? = null,
    val openingSpotlightLine: String? = null,
    val openingSpotlightAlternates: JSONArray = JSONArray(),
    val scanArenaLine: String? = null,
    val pressureBeats: JSONArray = JSONArray(),
    val survivorRevealLine: String? = null,
    val boundedLandingLine: String? = null,
    val naturalReturnOfferLine: String? = null,
    val boundedFollowupOfferLine: String? = null,
    val closureStyleTag: String? = null,
    val orderedHouses: List<String> = emptyList(),
    val survivingDigits: List<Int> = emptyList(),
    val askedDigitSurvives: Boolean = false,
    val decisiveFact: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "LOCAL_PERMISSIBILITY_SCAN"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("target_cell", targetCell ?: JSONObject.NULL)
        put("asked_digit", askedDigit ?: JSONObject.NULL)

        put("opening_spotlight_line", openingSpotlightLine ?: JSONObject.NULL)
        put("opening_spotlight_alternates", openingSpotlightAlternates)
        put("scan_arena_line", scanArenaLine ?: JSONObject.NULL)
        put("pressure_beats", pressureBeats)
        put("survivor_reveal_line", survivorRevealLine ?: JSONObject.NULL)
        put("bounded_landing_line", boundedLandingLine ?: JSONObject.NULL)
        put("natural_return_offer_line", naturalReturnOfferLine ?: JSONObject.NULL)
        put("bounded_followup_offer_line", boundedFollowupOfferLine ?: JSONObject.NULL)
        put("closure_style_tag", closureStyleTag ?: JSONObject.NULL)
        put("ordered_houses", JSONArray().apply { orderedHouses.forEach { put(it) } })
        put("surviving_digits", JSONArray().apply { survivingDigits.forEach { put(it) } })
        put("asked_digit_survives", askedDigitSurvives)

        put("decisive_fact", decisiveFact ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class HouseAlreadyOccupiedAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val targetHouse: String? = null,
    val askedDigit: Int? = null,
    val existingDigitCell: String? = null,
    val openingFactLine: String? = null,
    val duplicateRuleLine: String? = null,
    val supportingSeatClosureLine: String? = null,
    val boundedLandingLine: String? = null,
    val naturalReturnOfferLine: String? = null,
    val boundedFollowupOfferLine: String? = null,
    val closureStyleTag: String? = null,
    val decisiveFact: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "HOUSE_ALREADY_OCCUPIED"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("target_house", targetHouse ?: JSONObject.NULL)
        put("asked_digit", askedDigit ?: JSONObject.NULL)
        put("existing_digit_cell", existingDigitCell ?: JSONObject.NULL)

        put("opening_fact_line", openingFactLine ?: JSONObject.NULL)
        put("duplicate_rule_line", duplicateRuleLine ?: JSONObject.NULL)
        put("supporting_seat_closure_line", supportingSeatClosureLine ?: JSONObject.NULL)
        put("bounded_landing_line", boundedLandingLine ?: JSONObject.NULL)
        put("natural_return_offer_line", naturalReturnOfferLine ?: JSONObject.NULL)
        put("bounded_followup_offer_line", boundedFollowupOfferLine ?: JSONObject.NULL)
        put("closure_style_tag", closureStyleTag ?: JSONObject.NULL)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class FilledCellFactAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val targetCell: String? = null,
    val askedDigit: Int? = null,
    val placedValue: Int? = null,
    val openingFactLine: String? = null,
    val occupancyClarifierLine: String? = null,
    val boundedLandingLine: String? = null,
    val naturalReturnOfferLine: String? = null,
    val boundedFollowupOfferLine: String? = null,
    val closureStyleTag: String? = null,
    val decisiveFact: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "CELL_ALREADY_FILLED"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("target_cell", targetCell ?: JSONObject.NULL)
        put("asked_digit", askedDigit ?: JSONObject.NULL)
        put("placed_value", placedValue ?: JSONObject.NULL)

        put("opening_fact_line", openingFactLine ?: JSONObject.NULL)
        put("occupancy_clarifier_line", occupancyClarifierLine ?: JSONObject.NULL)
        put("bounded_landing_line", boundedLandingLine ?: JSONObject.NULL)
        put("natural_return_offer_line", naturalReturnOfferLine ?: JSONObject.NULL)
        put("bounded_followup_offer_line", boundedFollowupOfferLine ?: JSONObject.NULL)
        put("closure_style_tag", closureStyleTag ?: JSONObject.NULL)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class SurvivorLadderAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.SURVIVOR_LADDER,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val targetCell: String? = null,
    val askedDigit: Int? = null,
    val ladderRows: JSONArray = JSONArray(),
    val survivorSummary: JSONObject = JSONObject(),
    val decisiveFact: String? = null,
    val allowedStageBoundary: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "SURVIVOR_LADDER"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("target_cell", targetCell ?: JSONObject.NULL)
        put("asked_digit", askedDigit ?: JSONObject.NULL)
        put("ladder_rows", ladderRows)
        put("survivor_summary", survivorSummary)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)
        put("allowed_stage_boundary", allowedStageBoundary ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class ContrastDuelAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.CONTRAST_DUEL,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val primaryCell: String? = null,
    val rivalCell: String? = null,
    val askedDigit: Int? = null,
    val contrastSummary: JSONObject = JSONObject(),
    val ladderRows: JSONArray = JSONArray(),
    val decisiveFact: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "CONTRAST_DUEL"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("primary_cell", primaryCell ?: JSONObject.NULL)
        put("rival_cell", rivalCell ?: JSONObject.NULL)
        put("asked_digit", askedDigit ?: JSONObject.NULL)
        put("contrast_summary", contrastSummary)
        put("ladder_rows", ladderRows)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class PatternLegitimacyAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.PATTERN_LEGITIMACY_CHECK,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val centralQuestion: String? = null,
    val claimedTechniqueId: String? = null,
    val legitimacySummary: JSONObject = JSONObject(),
    val ladderRows: JSONArray = JSONArray(),
    val decisiveFact: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "PATTERN_LEGITIMACY_CHECK"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("claimed_technique_id", claimedTechniqueId ?: JSONObject.NULL)
        put("legitimacy_summary", legitimacySummary)
        put("ladder_rows", ladderRows)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}

data class HonestInsufficiencyAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.HONEST_INSUFFICIENCY_ANSWER,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val claimKind: String? = null,
    val centralQuestion: String? = null,
    val directAnswer: String? = null,
    val nonproofReason: String? = null,
    val localStateSummary: JSONArray = JSONArray(),
    val decisiveFact: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "HONEST_INSUFFICIENCY_ANSWER"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("claim_kind", claimKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("direct_answer", directAnswer ?: JSONObject.NULL)
        put("nonproof_reason", nonproofReason ?: JSONObject.NULL)
        put("local_state_summary", localStateSummary)
        put("decisive_fact", decisiveFact ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}


/* -------------------------------------------------------------------------
 * Wave 1 atom 2 — state readout
 * ------------------------------------------------------------------------- */

data class StateReadoutAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.LOCAL_GRID_INSPECTION,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.STATE_READOUT,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val readoutKind: String? = null,
    val centralQuestion: String? = null,
    val stateSummary: JSONArray = JSONArray(),
    val whyItMatters: String? = null,

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_PROOF_LADDER,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "STATE_READOUT"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("readout_kind", readoutKind ?: JSONObject.NULL)
        put("central_question", centralQuestion ?: JSONObject.NULL)
        put("state_summary", stateSummary)
        put("why_it_matters", whyItMatters ?: JSONObject.NULL)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}


/* -------------------------------------------------------------------------
 * Wave 1 atom 3 — proposal verdict
 * ------------------------------------------------------------------------- */

data class ProposalVerdictAtomV1(
    override val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.USER_PROPOSAL_VERDICT,
    override val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.PROPOSAL_VERDICT,

    override val focusScope: String? = null,
    override val focusCells: List<String> = emptyList(),
    override val focusHouses: List<String> = emptyList(),

    val proposalSummary: String? = null,
    val verdict: String? = null,
    val whatIsCorrect: JSONArray = JSONArray(),
    val whatIsIncorrect: JSONArray = JSONArray(),
    val missingCondition: String? = null,
    val routeRelation: String? = null,
    val evidenceSummary: JSONArray = JSONArray(),

    override val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
    ),
    override val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
) : DetourNarrativeAtomV1 {

    override val atomKind: String = "PROPOSAL_VERDICT"

    override fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_atom_v1")
        put("atom_kind", atomKind)
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("focus_scope", focusScope ?: JSONObject.NULL)
        put("focus_cells", jsonArrayOfStringsV1(focusCells))
        put("focus_houses", jsonArrayOfStringsV1(focusHouses))

        put("proposal_summary", proposalSummary ?: JSONObject.NULL)
        put("verdict", verdict ?: JSONObject.NULL)
        put("what_is_correct", whatIsCorrect)
        put("what_is_incorrect", whatIsIncorrect)
        put("missing_condition", missingCondition ?: JSONObject.NULL)
        put("route_relation", routeRelation ?: JSONObject.NULL)
        put("evidence_summary", evidenceSummary)

        put("answer_boundary", jsonArrayOfEnumNamesV1(answerBoundary))
        put("handover_mode", handoverMode.name)
    }
}


/* -------------------------------------------------------------------------
 * Native detour narrative context
 * ------------------------------------------------------------------------- */

data class DetourNarrativeContextV1(
    val demandCategory: DetourDemandCategoryV2,
    val archetype: DetourNarrativeArchetypeV1,
    val dominantAtom: DetourNarrativeAtomV1,

    val anchorStepId: String? = null,
    val anchorStage: String? = null,
    val pausedRouteCheckpointId: String? = null,

    val overlayPolicy: DetourOverlayPolicyV1 = DetourOverlayPolicyV1(),
    val handbackPolicy: DetourHandbackPolicyV1 = DetourHandbackPolicyV1(),
    val visualContract: DetourVisualContractV1 = DetourVisualContractV1(),
    val narrativeSupport: JSONObject = JSONObject(),
    val debugSupport: JSONObject = JSONObject(),

    val createdTurnSeq: Long,
    val sourceQueryFamily: String? = null,
    val sourceQueryProfile: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_narrative_context_v1")
        put("demand_category", demandCategory.name)
        put("archetype", archetype.name)

        put("anchor_step_id", anchorStepId ?: JSONObject.NULL)
        put("anchor_stage", anchorStage ?: JSONObject.NULL)
        put("paused_route_checkpoint_id", pausedRouteCheckpointId ?: JSONObject.NULL)

        put("overlay_policy", overlayPolicy.toJson())
        put("handback_policy", handbackPolicy.toJson())
        put("visual_contract", visualContract.toJson())
        put("narrative_support", narrativeSupport)
        put("debug_support", debugSupport)

        put("created_turn_seq", createdTurnSeq)
        put("source_query_family", sourceQueryFamily ?: JSONObject.NULL)
        put("source_query_profile", sourceQueryProfile ?: JSONObject.NULL)

        put("dominant_atom", dominantAtom.toJson())
    }
}