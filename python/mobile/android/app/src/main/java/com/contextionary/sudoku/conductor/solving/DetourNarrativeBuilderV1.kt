package com.contextionary.sudoku.conductor.solving

import com.contextionary.sudoku.conductor.policy.DetourAnswerBoundaryV1
import com.contextionary.sudoku.conductor.policy.DetourDemandCategoryV2
import com.contextionary.sudoku.conductor.policy.DetourHandbackPolicyV1
import com.contextionary.sudoku.conductor.policy.DetourHandoverModeV1
import com.contextionary.sudoku.conductor.policy.DetourNarrativeArchetypeV1
import com.contextionary.sudoku.conductor.policy.DetourOverlayModeV1
import com.contextionary.sudoku.conductor.policy.DetourOverlayPolicyV1
import org.json.JSONArray
import org.json.JSONObject

/**
 * DetourNarrativeBuilderV1
 *
 * Permanent-design native builders for detour narrative context.
 *
 * Input:
 * - canonical normalized detour truth emitted by normalize_detour.py
 *
 * Output:
 * - native DetourNarrativeContextV1 with one dominant typed atom
 *
 * Important:
 * - this builder is a sibling of the packet builders from F7R
 * - it reads canonical normalized truth directly
 * - it does NOT reconstruct atoms from reply packets
 */
object DetourNarrativeBuilderV1 {

    private fun firstNonBlank(vararg values: String?): String? {
        values.forEach { raw ->
            val s = raw?.trim()
            if (!s.isNullOrEmpty()) return s
        }
        return null
    }

    private fun stringList(arr: JSONArray?): List<String> {
        if (arr == null) return emptyList()
        val out = mutableListOf<String>()
        for (i in 0 until arr.length()) {
            val s = arr.optString(i, "").trim()
            if (s.isNotEmpty()) out += s
        }
        return out
    }

    private fun copyArray(arr: JSONArray?): JSONArray {
        val out = JSONArray()
        if (arr == null) return out
        for (i in 0 until arr.length()) out.put(arr.get(i))
        return out
    }

    private fun spokenCellLabel(cell: String?): String? {
        val raw = cell?.trim() ?: return null
        val match = Regex("^r([1-9])c([1-9])$").matchEntire(raw) ?: return raw
        val row = match.groupValues[1]
        val col = match.groupValues[2]
        return "row $row, column $col"
    }

    private fun jsonStringArray(vararg values: String?): JSONArray =
        JSONArray().apply {
            values.forEach { v ->
                val s = v?.trim()
                if (!s.isNullOrEmpty()) put(s)
            }
        }

    private fun authoredReturnOfferLine(
        targetCell: String?,
        askedDigit: Int?,
        handbackPolicy: DetourHandbackPolicyV1
    ): String? {
        if (handbackPolicy.handoverMode != DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE &&
            handbackPolicy.handoverMode != DetourHandoverModeV1.RETURN_TO_CURRENT_STAGE
        ) return null

        val cellLabel = spokenCellLabel(targetCell)
        return when {
            !cellLabel.isNullOrBlank() && askedDigit != null ->
                "So $askedDigit stays alive in $cellLabel for now. We can step back into the move whenever you’re ready."
            !cellLabel.isNullOrBlank() ->
                "So that square stays open for now. We can step back into the move whenever you’re ready."
            else ->
                "So the local question stays open for now. We can step back into the move whenever you’re ready."
        }
    }

    private fun authoredBoundedFollowupOfferLine(
        targetCell: String?,
        askedDigit: Int?
    ): String? {
        val cellLabel = spokenCellLabel(targetCell)
        return when {
            !cellLabel.isNullOrBlank() && askedDigit != null ->
                "Or, if you want, we can stay with $cellLabel a moment longer and test another candidate there."
            !cellLabel.isNullOrBlank() ->
                "Or, if you want, we can stay with that square a moment longer and test one more local question."
            else ->
                "Or, if you want, we can stay with this local spot a moment longer and test one more question."
        }
    }

    private fun overlayMode(raw: String?): DetourOverlayModeV1 =
        when (raw?.trim()?.uppercase()) {
            "REPLACE" -> DetourOverlayModeV1.REPLACE
            "CLEAR" -> DetourOverlayModeV1.CLEAR
            "PRESERVE" -> DetourOverlayModeV1.PRESERVE
            else -> DetourOverlayModeV1.AUGMENT
        }

    private fun handoverMode(raw: String?): DetourHandoverModeV1 =
        when (raw?.trim()?.uppercase()) {
            "RETURN_TO_CURRENT_STAGE" -> DetourHandoverModeV1.RETURN_TO_CURRENT_STAGE
            "HOLD_FOCUS_HERE" -> DetourHandoverModeV1.HOLD_FOCUS_HERE
            "AWAIT_USER_CONTROL" -> DetourHandoverModeV1.AWAIT_USER_CONTROL
            "REPAIR_THEN_RETURN" -> DetourHandoverModeV1.REPAIR_THEN_RETURN
            else -> DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
        }

    private fun overlayPolicyFromNormalized(normalized: JSONObject): DetourOverlayPolicyV1 {
        val overlay = normalized.optJSONObject("overlay_context") ?: JSONObject()
        val overlayStoryKind = firstNonBlank(overlay.optString("overlay_story_kind", null))
        return DetourOverlayPolicyV1(
            overlayMode = overlayMode(overlay.optString("overlay_mode", null)),
            primaryFocusCells = stringList(overlay.optJSONArray("focus_cells")),
            primaryFocusHouses = stringList(overlay.optJSONArray("focus_houses")),
            secondaryFocusCells = stringList(overlay.optJSONArray("secondary_focus_cells")),
            deemphasizeCells = stringList(overlay.optJSONArray("deemphasize_cells")),
            reasonForFocus = firstNonBlank(
                overlayStoryKind?.let { "move_proof:$it" },
                overlay.optString("reason_for_focus", null)
            ),
            expectedSpokenAnchor = firstNonBlank(
                overlayStoryKind,
                overlay.optString("reason_for_focus", null),
                overlay.optJSONArray("focus_cells")?.optString(0, null),
                overlay.optJSONArray("focus_houses")?.optString(0, null)
            )
        )
    }


    private fun distinctNonBlankStrings(vararg groups: List<String>): List<String> =
        buildList {
            groups.forEach { group ->
                group.forEach { raw ->
                    val s = raw.trim()
                    if (s.isNotEmpty() && !contains(s)) add(s)
                }
            }
        }

    private fun addVisualCellIfPresent(
        out: MutableList<DetourVisualCellV1>,
        cellRef: String?,
        role: String,
        digit: Int? = null,
        emphasis: String? = null,
        candidateDigits: List<Int> = emptyList(),
        meta: JSONObject = JSONObject()
    ) {
        val cell = cellRef?.trim()?.takeIf { it.isNotEmpty() } ?: return
        if (out.any { it.cellRef == cell && it.role == role && it.digit == digit }) return
        out += DetourVisualCellV1(
            cellRef = cell,
            role = role,
            digit = digit,
            emphasis = emphasis,
            candidateDigits = candidateDigits,
            meta = meta
        )
    }

    private fun addVisualHouseIfPresent(
        out: MutableList<DetourVisualHouseV1>,
        houseRef: String?,
        role: String,
        emphasis: String? = null,
        meta: JSONObject = JSONObject()
    ) {
        val house = houseRef?.trim()?.takeIf { it.isNotEmpty() } ?: return
        if (out.any { it.houseRef == house && it.role == role }) return
        out += DetourVisualHouseV1(
            houseRef = house,
            role = role,
            emphasis = emphasis,
            meta = meta
        )
    }

    private fun addVisualCandidateIfPresent(
        out: MutableList<DetourVisualCandidateMarkV1>,
        cellRef: String?,
        digit: Int?,
        role: String,
        emphasis: String? = null,
        meta: JSONObject = JSONObject()
    ) {
        val cell = cellRef?.trim()?.takeIf { it.isNotEmpty() } ?: return
        val d = digit?.takeIf { it in 1..9 } ?: return
        if (out.any { it.cellRef == cell && it.digit == d && it.role == role }) return
        out += DetourVisualCandidateMarkV1(
            cellRef = cell,
            digit = d,
            role = role,
            emphasis = emphasis,
            meta = meta
        )
    }

    private fun moveProofStoryKindFromArchetypeV1(
        archetype: DetourNarrativeArchetypeV1
    ): String? =
        when (archetype) {
            DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT -> "CONTRADICTION_SPOTLIGHT"
            DetourNarrativeArchetypeV1.SURVIVOR_LADDER -> "SURVIVOR_LADDER"
            DetourNarrativeArchetypeV1.CONTRAST_DUEL -> "CONTRAST_DUEL"
            DetourNarrativeArchetypeV1.PATTERN_LEGITIMACY_CHECK -> "PATTERN_LEGITIMACY"
            DetourNarrativeArchetypeV1.HONEST_INSUFFICIENCY_ANSWER -> "HONEST_INSUFFICIENCY"
            DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN -> "LOCAL_PERMISSIBILITY_SCAN"
            DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED -> "HOUSE_ALREADY_OCCUPIED"
            DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED -> "CELL_ALREADY_FILLED"
            DetourNarrativeArchetypeV1.LOCAL_PROOF_SPOTLIGHT -> "LOCAL_PROOF_SPOTLIGHT"
            else -> null
        }

    private fun preferredVisualFocusCellV1(
        atom: DetourNarrativeAtomV1,
        overlayPolicy: DetourOverlayPolicyV1
    ): String? {
        overlayPolicy.primaryFocusCells.firstOrNull()?.let { return it }

        return when (atom) {
            is ContradictionSpotlightAtomV1 ->
                firstNonBlank(atom.targetCell, atom.blockerCell, atom.focusCells.firstOrNull())

            is SurvivorLadderAtomV1 ->
                firstNonBlank(atom.targetCell, atom.focusCells.firstOrNull())

            is ContrastDuelAtomV1 ->
                firstNonBlank(atom.primaryCell, atom.rivalCell, atom.focusCells.firstOrNull())

            is LocalPermissibilityScanAtomV1 ->
                firstNonBlank(atom.targetCell, atom.focusCells.firstOrNull())

            is HouseAlreadyOccupiedAtomV1 ->
                firstNonBlank(atom.existingDigitCell, atom.focusCells.firstOrNull())

            is FilledCellFactAtomV1 ->
                firstNonBlank(atom.targetCell, atom.focusCells.firstOrNull())

            else ->
                atom.focusCells.firstOrNull()
        }
    }

    private fun buildMoveProofVisualContractV1(
        archetype: DetourNarrativeArchetypeV1,
        atom: DetourNarrativeAtomV1,
        overlayPolicy: DetourOverlayPolicyV1
    ): DetourVisualContractV1 {
        val primaryCells = mutableListOf<DetourVisualCellV1>()
        val secondaryCells = mutableListOf<DetourVisualCellV1>()
        val houses = mutableListOf<DetourVisualHouseV1>()
        val candidateMarks = mutableListOf<DetourVisualCandidateMarkV1>()

        overlayPolicy.primaryFocusCells.forEach { cell ->
            addVisualCellIfPresent(primaryCells, cell, role = "focus", emphasis = "primary")
        }
        overlayPolicy.secondaryFocusCells.forEach { cell ->
            addVisualCellIfPresent(secondaryCells, cell, role = "witness", emphasis = "secondary")
        }
        overlayPolicy.primaryFocusHouses.forEach { house ->
            addVisualHouseIfPresent(houses, house, role = "arena", emphasis = "primary")
        }

        when (atom) {
            is ContradictionSpotlightAtomV1 -> {
                addVisualCellIfPresent(primaryCells, atom.targetCell, role = "focus", emphasis = "primary")
                addVisualCellIfPresent(secondaryCells, atom.blockerCell, role = "blocker", digit = atom.askedDigit, emphasis = "secondary")
                addVisualHouseIfPresent(houses, atom.blockerHouse, role = "blocking_house", emphasis = "primary")
                addVisualCandidateIfPresent(candidateMarks, atom.targetCell, atom.askedDigit, role = "blocked", emphasis = "primary")
            }

            is SurvivorLadderAtomV1 -> {
                addVisualCellIfPresent(primaryCells, atom.targetCell, role = "focus", emphasis = "primary")
                addVisualCandidateIfPresent(candidateMarks, atom.targetCell, atom.askedDigit, role = "candidate_focus", emphasis = "primary")
            }

            is ContrastDuelAtomV1 -> {
                addVisualCellIfPresent(primaryCells, atom.primaryCell, role = "focus", digit = atom.askedDigit, emphasis = "primary")
                addVisualCellIfPresent(secondaryCells, atom.rivalCell, role = "rival", digit = atom.askedDigit, emphasis = "secondary")
            }

            is PatternLegitimacyAtomV1 -> {
                atom.focusCells.forEach { cell ->
                    addVisualCellIfPresent(primaryCells, cell, role = "pattern_member", emphasis = "primary")
                }
                atom.focusHouses.forEach { house ->
                    addVisualHouseIfPresent(houses, house, role = "pattern_house", emphasis = "primary")
                }
            }

            is HonestInsufficiencyAtomV1 -> {
                atom.focusCells.forEach { cell ->
                    addVisualCellIfPresent(primaryCells, cell, role = "focus", emphasis = "primary")
                }
                atom.focusHouses.forEach { house ->
                    addVisualHouseIfPresent(houses, house, role = "arena", emphasis = "primary")
                }
            }

            is LocalPermissibilityScanAtomV1 -> {
                addVisualCellIfPresent(primaryCells, atom.targetCell, role = "focus", emphasis = "primary")
                addVisualCandidateIfPresent(candidateMarks, atom.targetCell, atom.askedDigit, role = "candidate_focus", emphasis = "primary")
                atom.orderedHouses.forEach { house ->
                    addVisualHouseIfPresent(houses, house, role = "scan_house", emphasis = "primary")
                }
            }

            is HouseAlreadyOccupiedAtomV1 -> {
                addVisualCellIfPresent(primaryCells, atom.existingDigitCell, role = "occupied", digit = atom.askedDigit, emphasis = "primary")
                addVisualHouseIfPresent(houses, atom.targetHouse, role = "target_house", emphasis = "primary")
            }

            is FilledCellFactAtomV1 -> {
                addVisualCellIfPresent(primaryCells, atom.targetCell, role = "occupied", digit = atom.placedValue, emphasis = "primary")
            }

            is LocalProofSpotlightAtomV1 -> {
                atom.focusCells.forEach { cell ->
                    addVisualCellIfPresent(primaryCells, cell, role = "focus", emphasis = "primary")
                }
                atom.focusHouses.forEach { house ->
                    addVisualHouseIfPresent(houses, house, role = "arena", emphasis = "primary")
                }
            }

            else -> {
                atom.focusCells.forEach { cell ->
                    addVisualCellIfPresent(primaryCells, cell, role = "focus", emphasis = "primary")
                }
                atom.focusHouses.forEach { house ->
                    addVisualHouseIfPresent(houses, house, role = "arena", emphasis = "primary")
                }
            }
        }

        val hasPrimary = primaryCells.isNotEmpty()
        val hasSecondary = secondaryCells.isNotEmpty()
        val hasHouses = houses.isNotEmpty()
        val hasDeemphasis = overlayPolicy.deemphasizeCells.isNotEmpty()

        val defaultLayers = linkedSetOf<String>()
        defaultLayers += "detour:native"

        when (moveProofStoryKindFromArchetypeV1(archetype)) {
            "CONTRADICTION_SPOTLIGHT" -> {
                if (hasPrimary) defaultLayers += "detour:primary"
                if (hasSecondary) defaultLayers += "detour:secondary"
                if (hasHouses) defaultLayers += "detour:houses"
            }

            "SURVIVOR_LADDER" -> {
                if (hasPrimary) defaultLayers += "detour:primary"
                if (hasHouses) defaultLayers += "detour:houses"
                if (hasSecondary) defaultLayers += "detour:secondary"
            }

            "CONTRAST_DUEL" -> {
                if (hasPrimary) defaultLayers += "detour:primary"
                if (hasSecondary) defaultLayers += "detour:secondary"
                if (hasHouses) defaultLayers += "detour:houses"
            }

            "PATTERN_LEGITIMACY" -> {
                if (hasPrimary) defaultLayers += "detour:primary"
                if (hasHouses) defaultLayers += "detour:houses"
                if (hasSecondary) defaultLayers += "detour:secondary"
            }

            "HONEST_INSUFFICIENCY" -> {
                if (hasPrimary) defaultLayers += "detour:primary"
                if (hasHouses) defaultLayers += "detour:houses"
            }

            else -> {
                if (hasPrimary) defaultLayers += "detour:primary"
                if (hasSecondary) defaultLayers += "detour:secondary"
                if (hasHouses) defaultLayers += "detour:houses"
            }
        }

        if (hasDeemphasis) defaultLayers += "detour:deemphasize"

        val availableLayers = linkedSetOf<String>()
        availableLayers += "detour:native"
        if (hasPrimary) availableLayers += "detour:primary"
        if (hasSecondary) availableLayers += "detour:secondary"
        if (hasHouses) availableLayers += "detour:houses"
        if (hasDeemphasis) availableLayers += "detour:deemphasize"

        return DetourVisualContractV1(
            storyKind = moveProofStoryKindFromArchetypeV1(archetype),
            focusCell = preferredVisualFocusCellV1(atom, overlayPolicy),
            primaryCells = primaryCells,
            secondaryCells = secondaryCells,
            houses = houses,
            candidateMarks = candidateMarks,
            deemphasizeCells = overlayPolicy.deemphasizeCells,
            availableLayers = availableLayers.toList(),
            defaultAppliedLayers = defaultLayers.toList(),
            sceneNotes = JSONObject().apply {
                put("wave", "wave_1")
                put("builder_authored", true)
                put("constitution_mode", "seed_contract")
            }
        )
    }

    private fun buildLocalInspectionVisualContractV1(
        atom: StateReadoutAtomV1,
        overlayPolicy: DetourOverlayPolicyV1
    ): DetourVisualContractV1 {
        val primaryCells = mutableListOf<DetourVisualCellV1>()
        val houses = mutableListOf<DetourVisualHouseV1>()

        distinctNonBlankStrings(
            overlayPolicy.primaryFocusCells,
            atom.focusCells
        ).forEach { cell ->
            addVisualCellIfPresent(primaryCells, cell, role = "focus", emphasis = "primary")
        }

        distinctNonBlankStrings(
            overlayPolicy.primaryFocusHouses,
            atom.focusHouses
        ).forEach { house ->
            addVisualHouseIfPresent(houses, house, role = "arena", emphasis = "primary")
        }

        val availableLayers = buildList {
            add("detour:native")
            if (primaryCells.isNotEmpty()) add("detour:primary")
            if (houses.isNotEmpty()) add("detour:houses")
            if (overlayPolicy.deemphasizeCells.isNotEmpty()) add("detour:deemphasize")
        }

        return DetourVisualContractV1(
            storyKind = "STATE_READOUT",
            focusCell = primaryCells.firstOrNull()?.cellRef,
            primaryCells = primaryCells,
            houses = houses,
            deemphasizeCells = overlayPolicy.deemphasizeCells,
            availableLayers = availableLayers,
            defaultAppliedLayers = availableLayers,
            sceneNotes = JSONObject().apply {
                put("wave", "wave_1")
                put("builder_authored", true)
                put("constitution_mode", "seed_contract")
            }
        )
    }

    private fun buildProposalVerdictVisualContractV1(
        atom: ProposalVerdictAtomV1,
        overlayPolicy: DetourOverlayPolicyV1
    ): DetourVisualContractV1 {
        val primaryCells = mutableListOf<DetourVisualCellV1>()
        val houses = mutableListOf<DetourVisualHouseV1>()

        distinctNonBlankStrings(
            overlayPolicy.primaryFocusCells,
            atom.focusCells
        ).forEach { cell ->
            addVisualCellIfPresent(primaryCells, cell, role = "focus", emphasis = "primary")
        }

        distinctNonBlankStrings(
            overlayPolicy.primaryFocusHouses,
            atom.focusHouses
        ).forEach { house ->
            addVisualHouseIfPresent(houses, house, role = "arena", emphasis = "primary")
        }

        val availableLayers = buildList {
            add("detour:native")
            if (primaryCells.isNotEmpty()) add("detour:primary")
            if (houses.isNotEmpty()) add("detour:houses")
            if (overlayPolicy.deemphasizeCells.isNotEmpty()) add("detour:deemphasize")
        }

        return DetourVisualContractV1(
            storyKind = "PROPOSAL_VERDICT",
            focusCell = primaryCells.firstOrNull()?.cellRef,
            primaryCells = primaryCells,
            houses = houses,
            deemphasizeCells = overlayPolicy.deemphasizeCells,
            availableLayers = availableLayers,
            defaultAppliedLayers = availableLayers,
            sceneNotes = JSONObject().apply {
                put("wave", "wave_1")
                put("builder_authored", true)
                put("constitution_mode", "seed_contract")
            }
        )
    }

    private fun handbackPolicyFromNormalized(normalized: JSONObject): DetourHandbackPolicyV1 {
        val anchor = normalized.optJSONObject("anchor") ?: JSONObject()
        val route = normalized.optJSONObject("route_context") ?: JSONObject()
        val mode = handoverMode(route.optString("recommended_handover_mode", null))
        return DetourHandbackPolicyV1(
            handoverMode = mode,
            pausedRouteCheckpoint = firstNonBlank(anchor.optString("paused_route_checkpoint_id", null)),
            returnTargetStage = firstNonBlank(anchor.optString("story_stage", null)),
            returnTargetStepId = firstNonBlank(anchor.optString("step_id", null)),
            stayDetachedUntilUserSaysContinue = mode == DetourHandoverModeV1.AWAIT_USER_CONTROL,
            spokenReturnLine = firstNonBlank(route.optString("spoken_return_line", null))
        )
    }


    private fun moveProofArchetypeFromNormalized(normalized: JSONObject, proofTruth: JSONObject): DetourNarrativeArchetypeV1 {
        val raw = firstNonBlank(
            normalized.optString("narrative_archetype", null),
            proofTruth.optString("narrative_archetype", null)
        )?.trim()?.uppercase()

        val question = normalized.optJSONObject("question") ?: JSONObject()
        val methodFamily = firstNonBlank(
            normalized.optString("method_family", null),
            proofTruth.optString("method_family", null)
        )?.trim()?.uppercase()
        val proofObject = firstNonBlank(
            proofTruth.optString("proof_object", null),
            proofTruth.optString("claim_kind", null)
        )?.trim()?.uppercase()
        val challengeLane = firstNonBlank(
            normalized.optString("challenge_lane", null),
            proofTruth.optString("challenge_lane", null)
        )?.trim()?.uppercase()
        val answerPolarity = firstNonBlank(
            proofTruth.optString("answer_polarity", null)
        )?.trim()?.uppercase()
        val rivalCell = firstNonBlank(question.optString("rival_cell", null))
        val claimedTechniqueId = firstNonBlank(question.optString("claimed_technique_id", null))
        val geometryKind = firstNonBlank(
            proofTruth.optJSONObject("local_proof_geometry")?.optString("geometry_kind", null),
            normalized.optJSONObject("local_proof_geometry")?.optString("geometry_kind", null)
        )?.trim()?.uppercase()

        val inferred = when {
            proofObject == "HOUSE_ALREADY_CONTAINS_DIGIT" ||
                    geometryKind == "HOUSE_DIGIT_ALREADY_PLACED" ||
                    answerPolarity == "ALREADY_PLACED" ->
                DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED

            proofObject == "CELL_ALREADY_FILLED" ||
                    geometryKind == "CELL_ALREADY_FILLED" ||
                    answerPolarity == "ALREADY_FILLED" ->
                DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED

            methodFamily == "TECHNIQUE_LEGITIMACY" ||
                    challengeLane == "TECHNIQUE_LEGITIMACY" ||
                    geometryKind == "PATTERN_STRUCTURE_FRAME" ||
                    !claimedTechniqueId.isNullOrBlank() ->
                DetourNarrativeArchetypeV1.PATTERN_LEGITIMACY_CHECK

            methodFamily == "CONTRAST_TEST" ||
                    geometryKind == "RIVAL_COMPARISON_FRAME" ||
                    proofObject == "CELL_A_WINS_OVER_CELL_B_FOR_DIGIT" ||
                    !rivalCell.isNullOrBlank() ->
                DetourNarrativeArchetypeV1.CONTRAST_DUEL

            geometryKind == "HOUSE_DIGIT_SEAT_MAP" ->
                DetourNarrativeArchetypeV1.SURVIVOR_LADDER

            geometryKind == "CELL_THREE_HOUSE_UNIVERSE" &&
                    answerPolarity == "NOT_LOCALLY_PROVED" &&
                    challengeLane == "ELIMINATION_LEGITIMACY" ->
                DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN

            methodFamily == "HOUSE_UNIQUENESS" ||
                    methodFamily == "RIVAL_ELIMINATION_LADDER" ||
                    proofObject == "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE" ||
                    proofObject == "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL" ||
                    proofObject == "CELL_CAN_BE_DIGIT" ->
                DetourNarrativeArchetypeV1.SURVIVOR_LADDER

            methodFamily == "DIRECT_CONTRADICTION" ||
                    methodFamily == "ACTION_LEGITIMACY" ||
                    proofObject == "CELL_CANNOT_BE_DIGIT" ||
                    proofObject == "HOUSE_BLOCKS_DIGIT_FOR_TARGET" ||
                    proofObject == "ELIMINATION_IS_LEGAL" ->
                DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT

            answerPolarity == "NOT_LOCALLY_PROVED" ->
                DetourNarrativeArchetypeV1.HONEST_INSUFFICIENCY_ANSWER

            challengeLane == "FORCEDNESS_OR_UNIQUENESS" ->
                DetourNarrativeArchetypeV1.SURVIVOR_LADDER

            challengeLane == "CANDIDATE_IMPOSSIBILITY" ||
                    challengeLane == "ELIMINATION_LEGITIMACY" ->
                DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT

            else ->
                DetourNarrativeArchetypeV1.LOCAL_PROOF_SPOTLIGHT
        }

        return when (raw) {
            "HOUSE_ALREADY_OCCUPIED" -> DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED
            "CELL_ALREADY_FILLED" -> DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED
            "LOCAL_CONTRADICTION_SPOTLIGHT" -> DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT
            "LOCAL_PERMISSIBILITY_SCAN" -> DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN
            "SURVIVOR_LADDER" -> DetourNarrativeArchetypeV1.SURVIVOR_LADDER
            "CONTRAST_DUEL" -> DetourNarrativeArchetypeV1.CONTRAST_DUEL
            "PATTERN_LEGITIMACY_CHECK" -> DetourNarrativeArchetypeV1.PATTERN_LEGITIMACY_CHECK
            "HONEST_INSUFFICIENCY_ANSWER" -> DetourNarrativeArchetypeV1.HONEST_INSUFFICIENCY_ANSWER
            else -> inferred
        }
    }



    // Series-E P5:
    // Proof-challenge detours no longer synthesize a canned spoken transition line.
    // Opening shape is now authored from doctrine + packet truth instead.


    private fun moveProofEvidenceSummary(
        proofTruth: JSONObject,
        proofLadder: JSONObject
    ): JSONArray {
        val ladderRows = proofLadder.optJSONArray("rows")
        if (ladderRows != null && ladderRows.length() > 0) return copyArray(ladderRows)
        return copyArray(
            proofTruth.optJSONArray("bounded_proof_rows")
                ?: proofTruth.optJSONArray("witness_rows")
        )
    }

    private fun moveProofDecisiveFact(
        answerTruth: JSONObject,
        proofTruth: JSONObject
    ): String? {
        return firstNonBlank(
            answerTruth.optString("one_sentence_claim", null),
            answerTruth.optString("short_answer", null),
            proofTruth.optString("decisive_fact", null),
            proofTruth.optString("proof_claim", null)
        )
    }

    private fun firstBlockerFromGeometry(localProofGeometry: JSONObject): Pair<String?, String?> {
        val receipts = localProofGeometry.optJSONArray("blocker_receipts") ?: JSONArray()
        for (i in 0 until receipts.length()) {
            val row = receipts.optJSONObject(i) ?: continue
            val blockerCell = firstNonBlank(row.optString("blocker_cell", null))
            val blockerHouse = firstNonBlank(row.optString("blocking_house", null))
            if (!blockerCell.isNullOrBlank() || !blockerHouse.isNullOrBlank()) {
                return blockerCell to blockerHouse
            }
        }
        return null to null
    }

    private fun survivingDigitsFromGeometry(localProofGeometry: JSONObject): List<Int> {
        val direct = localProofGeometry.optJSONArray("surviving_digits")
        if (direct != null && direct.length() > 0) {
            val out = mutableListOf<Int>()
            for (i in 0 until direct.length()) {
                val d = direct.optInt(i, -1)
                if (d in 1..9) out += d
            }
            return out.distinct().sorted()
        }

        val candidateStatusMap = localProofGeometry.optJSONArray("candidate_status_map") ?: JSONArray()
        val out = mutableListOf<Int>()
        for (i in 0 until candidateStatusMap.length()) {
            val row = candidateStatusMap.optJSONObject(i) ?: continue
            val digit = row.optInt("digit", -1)
            val status = row.optString("status", "").trim().uppercase()
            if (digit in 1..9 && status == "SURVIVES") out += digit
        }
        return out.distinct().sorted()
    }

    private fun orderedHousesFromStoryArcOrGeometry(
        storyArc: JSONObject,
        localProofGeometry: JSONObject,
        focusHouses: List<String>
    ): List<String> {
        val ordered = linkedSetOf<String>()

        val storyArcOrdered = storyArc.optJSONArray("ordered_houses")
        if (storyArcOrdered != null) {
            for (i in 0 until storyArcOrdered.length()) {
                val house = storyArcOrdered.optString(i, "").trim()
                if (house.isNotEmpty()) ordered += house
            }
        }

        val geometryHouses = localProofGeometry.optJSONArray("houses")
        if (geometryHouses != null) {
            for (i in 0 until geometryHouses.length()) {
                val house = geometryHouses.optString(i, "").trim()
                if (house.isNotEmpty()) ordered += house
            }
        }

        focusHouses.forEach { if (it.isNotBlank()) ordered += it }
        return ordered.toList()
    }

    private fun pressureBeatsFromGeometry(
        localProofGeometry: JSONObject,
        orderedHouses: List<String>,
        targetCell: String?,
        askedDigit: Int?
    ): JSONArray {
        val byHouse = localProofGeometry.optJSONObject("blocked_digits_by_house") ?: JSONObject()
        val out = JSONArray()

        orderedHouses.forEach { house ->
            val arr = byHouse.optJSONArray(house) ?: JSONArray()
            val blockedDigits = mutableListOf<Int>()
            val witnessCells = linkedSetOf<String>()

            for (i in 0 until arr.length()) {
                val receipt = arr.optJSONObject(i)
                if (receipt != null) {
                    val digit = receipt.optInt("digit", -1)
                    if (digit in 1..9) blockedDigits += digit
                    val blockerCell = firstNonBlank(receipt.optString("blocker_cell", null))
                    if (!blockerCell.isNullOrBlank()) witnessCells += blockerCell
                } else {
                    val digit = arr.optInt(i, -1)
                    if (digit in 1..9) blockedDigits += digit
                }
            }

            val dedupedDigits = blockedDigits.distinct().sorted()
            val spokenLine =
                when {
                    dedupedDigits.isNotEmpty() ->
                        buildString {
                            append(house)
                            append(" pushes out ")
                            append(dedupedDigits.joinToString(", "))
                            if (targetCell != null) {
                                append(" around ")
                                append(targetCell)
                            }
                            if (witnessCells.isNotEmpty()) {
                                append(", with pressure coming from ")
                                append(witnessCells.joinToString(", "))
                            }
                            append(".")
                        }
                    askedDigit in 1..9 ->
                        "$house adds no direct blocker against $askedDigit."
                    else ->
                        "$house adds no direct blocker in the local scan."
                }

            out.put(
                JSONObject().apply {
                    put("house_ref", house)
                    put("blocked_digits", JSONArray().apply { dedupedDigits.forEach { put(it) } })
                    put("witness_cells", JSONArray().apply { witnessCells.forEach { put(it) } })
                    put("spoken_line", spokenLine)
                }
            )
        }

        return out
    }


    private fun houseAlreadyOccupiedNarrativeSupport(
        localProofGeometry: JSONObject,
        focusHouses: List<String>,
        askedDigit: Int?,
        decisiveFact: String?,
        handbackPolicy: DetourHandbackPolicyV1
    ): JSONObject {
        val targetHouse = firstNonBlank(
            localProofGeometry.optString("target_house", null),
            focusHouses.firstOrNull()
        )
        val existingDigitCell = firstNonBlank(localProofGeometry.optString("existing_digit_cell", null))
        val houseLabel = targetHouse ?: "this house"
        val cellLabel = spokenCellLabel(existingDigitCell) ?: existingDigitCell

        val openSeatRows = localProofGeometry.optJSONArray("open_seat_rows") ?: JSONArray()
        val remainingSeats = mutableListOf<String>()
        for (i in 0 until openSeatRows.length()) {
            val row = openSeatRows.optJSONObject(i) ?: continue
            val seat = firstNonBlank(row.optString("seat", null))
            if (!seat.isNullOrBlank()) remainingSeats += (spokenCellLabel(seat) ?: seat)
        }

        val openingFactLine =
            when {
                askedDigit != null && !cellLabel.isNullOrBlank() && !houseLabel.isNullOrBlank() ->
                    "Let’s check $houseLabel first at the house level. The key fact is that $askedDigit is already sitting at $cellLabel."
                askedDigit != null && !houseLabel.isNullOrBlank() ->
                    "Let’s check $houseLabel first at the house level. The key fact is that $askedDigit is already placed there."
                else ->
                    "Let’s check this house first at the house level. The key fact is that the asked digit is already placed there."
            }

        val duplicateRuleLine =
            when {
                askedDigit != null && !houseLabel.isNullOrBlank() ->
                    "So $houseLabel does not still need a home for $askedDigit — it already has one."
                askedDigit != null ->
                    "So this house does not still need a home for $askedDigit — it already has one."
                else ->
                    "So this house does not still need another home for that digit — it already has one."
            }

        val supportingSeatClosureLine =
            when {
                askedDigit != null && remainingSeats.isNotEmpty() ->
                    "And that is exactly why the remaining open seats cannot take another $askedDigit: ${remainingSeats.joinToString(", ")}."
                askedDigit != null ->
                    "And that is exactly why no other open seat in the house can take another $askedDigit."
                else ->
                    "And that is exactly why no other open seat in the house can take that digit."
            }

        val boundedLandingLine =
            firstNonBlank(
                decisiveFact,
                if (askedDigit != null && !houseLabel.isNullOrBlank())
                    "So this is not a “where can $askedDigit go?” story. It is a “$askedDigit is already placed in $houseLabel” story."
                else
                    "So this is not an open seat-search story. It is an already-placed story."
            )

        return JSONObject().apply {
            put("target_house", targetHouse ?: JSONObject.NULL)
            put("existing_digit_cell", existingDigitCell ?: JSONObject.NULL)
            put("opening_fact_line", openingFactLine)
            put("duplicate_rule_line", duplicateRuleLine)
            put("supporting_seat_closure_line", supportingSeatClosureLine)
            put("bounded_landing_line", boundedLandingLine ?: JSONObject.NULL)
            put("natural_return_offer_line", authoredReturnOfferLine(existingDigitCell, askedDigit, handbackPolicy) ?: JSONObject.NULL)
            put("bounded_followup_offer_line", authoredBoundedFollowupOfferLine(existingDigitCell, askedDigit) ?: JSONObject.NULL)
            put("closure_style_tag", "HOUSE_FACT_THEN_DUPLICATE_RULE")
        }
    }

    private fun filledCellNarrativeSupport(
        localProofGeometry: JSONObject,
        targetCell: String?,
        askedDigit: Int?,
        decisiveFact: String?,
        handbackPolicy: DetourHandbackPolicyV1
    ): JSONObject {
        val cellRef = firstNonBlank(localProofGeometry.optString("target_cell", null), targetCell)
        val cellLabel = spokenCellLabel(cellRef) ?: cellRef
        val placedValue = localProofGeometry.optJSONObject("filled_state")?.optInt("placed_value", -1)
            ?.takeIf { it in 1..9 }
            ?: localProofGeometry.optInt("placed_value", -1).takeIf { it in 1..9 }

        val openingFactLine =
            when {
                !cellLabel.isNullOrBlank() && placedValue != null ->
                    "Let’s look at $cellLabel first. That cell is already filled with $placedValue."
                !cellLabel.isNullOrBlank() ->
                    "Let’s look at $cellLabel first. That cell is already filled."
                else ->
                    "Let’s look at that square first. It is already filled."
            }

        val occupancyClarifierLine =
            when {
                !cellLabel.isNullOrBlank() && placedValue != null && askedDigit != null && askedDigit == placedValue ->
                    "$askedDigit is not merely still possible there — it is already the placed value in $cellLabel."
                !cellLabel.isNullOrBlank() && placedValue != null && askedDigit != null ->
                    "So this is no longer a live candidate question: once $cellLabel is filled with $placedValue, it is not an open seat for testing $askedDigit."
                !cellLabel.isNullOrBlank() && placedValue != null ->
                    "So this is no longer a live candidate question: $cellLabel is already occupied by $placedValue."
                else ->
                    "So this is no longer a live candidate question because that square is already occupied."
            }

        val boundedLandingLine =
            firstNonBlank(
                decisiveFact,
                if (!cellLabel.isNullOrBlank() && placedValue != null)
                    "This is an already-filled-cell story, not an open candidate-seat story."
                else
                    "This is an already-filled-cell story."
            )

        return JSONObject().apply {
            put("target_cell", cellRef ?: JSONObject.NULL)
            put("placed_value", placedValue ?: JSONObject.NULL)
            put("opening_fact_line", openingFactLine)
            put("occupancy_clarifier_line", occupancyClarifierLine)
            put("bounded_landing_line", boundedLandingLine ?: JSONObject.NULL)
            put("natural_return_offer_line", authoredReturnOfferLine(cellRef, askedDigit, handbackPolicy) ?: JSONObject.NULL)
            put("bounded_followup_offer_line", authoredBoundedFollowupOfferLine(cellRef, askedDigit) ?: JSONObject.NULL)
            put("closure_style_tag", "FILLED_CELL_FACT_THEN_SCOPE_CLARIFIER")
        }
    }


    private fun localPermissibilityNarrativeSupport(
        storyArc: JSONObject,
        localProofGeometry: JSONObject,
        targetCell: String?,
        askedDigit: Int?,
        focusHouses: List<String>,
        decisiveFact: String?,
        handbackPolicy: DetourHandbackPolicyV1
    ): JSONObject {
        val orderedHouses = orderedHousesFromStoryArcOrGeometry(
            storyArc = storyArc,
            localProofGeometry = localProofGeometry,
            focusHouses = focusHouses
        )
        val survivingDigits = survivingDigitsFromGeometry(localProofGeometry)
        val askedDigitSurvives = askedDigit != null && survivingDigits.contains(askedDigit)
        val pressureBeats = pressureBeatsFromGeometry(
            localProofGeometry = localProofGeometry,
            orderedHouses = orderedHouses,
            targetCell = targetCell,
            askedDigit = askedDigit
        )

        val cellLabel = spokenCellLabel(targetCell) ?: targetCell

        val openingSpotlightLine =
            when {
                !cellLabel.isNullOrBlank() && askedDigit != null ->
                    "Let’s put $cellLabel under the spotlight and test whether $askedDigit really gets pushed out."
                !cellLabel.isNullOrBlank() ->
                    "Let’s put $cellLabel under the spotlight and test the local doubt carefully."
                else ->
                    "Let’s put this target cell under the spotlight and test the local doubt carefully."
            }

        val openingSpotlightAlternates =
            when {
                !cellLabel.isNullOrBlank() && askedDigit != null ->
                    jsonStringArray(
                        "Let’s focus tightly on $cellLabel and see whether $askedDigit really gets ruled out.",
                        "Now let’s narrow the beam to $cellLabel and ask the local question properly: does $askedDigit actually fail here?",
                        "Take $cellLabel as the scene for a moment. The real question is whether anything local truly knocks out $askedDigit."
                    )
                !cellLabel.isNullOrBlank() ->
                    jsonStringArray(
                        "Let’s focus tightly on $cellLabel and test the local picture.",
                        "Now let’s narrow the beam to $cellLabel and ask the local question properly.",
                        "Take $cellLabel as the scene for a moment and let the local evidence speak."
                    )
                else ->
                    jsonStringArray(
                        "Let’s focus tightly on this square and test the local picture.",
                        "Now let’s narrow the beam to this spot and ask the local question properly.",
                        "Take this square as the scene for a moment and let the local evidence speak."
                    )
            }

        val scanArenaLine =
            firstNonBlank(
                storyArc.optString("scan_arena_line", null),
                if (!cellLabel.isNullOrBlank())
                    "We let the three local judges speak in order: the row, the column, and the box around $cellLabel."
                else
                    "We let the three local judges speak in order: row, column, then box."
            )

        val survivorRevealLine =
            when {
                !cellLabel.isNullOrBlank() && askedDigit != null && askedDigitSurvives ->
                    "$askedDigit is still standing in $cellLabel after that local scan."
                survivingDigits.isNotEmpty() && !cellLabel.isNullOrBlank() ->
                    "$cellLabel still keeps ${survivingDigits.joinToString(", ")} alive after the local scan."
                askedDigit != null ->
                    "$askedDigit is not removed by any direct local blocker here."
                else ->
                    "The asked candidate still survives the local scan."
            }

        val boundedLandingLine =
            firstNonBlank(
                decisiveFact,
                if (!cellLabel.isNullOrBlank() && askedDigit != null)
                    "So there is no direct local reason to eliminate $askedDigit from $cellLabel."
                else
                    "So there is no direct local reason to eliminate the asked candidate here."
            )

        val naturalReturnOfferLine = authoredReturnOfferLine(
            targetCell = targetCell,
            askedDigit = askedDigit,
            handbackPolicy = handbackPolicy
        )

        val boundedFollowupOfferLine = authoredBoundedFollowupOfferLine(
            targetCell = targetCell,
            askedDigit = askedDigit
        )

        return JSONObject().apply {
            put("opening_spotlight_line", openingSpotlightLine)
            put("opening_spotlight_alternates", openingSpotlightAlternates)
            put("scan_arena_line", scanArenaLine ?: JSONObject.NULL)
            put("pressure_beats", pressureBeats)
            put("survivor_reveal_line", survivorRevealLine)
            put("bounded_landing_line", boundedLandingLine ?: JSONObject.NULL)
            put("natural_return_offer_line", naturalReturnOfferLine ?: JSONObject.NULL)
            put("bounded_followup_offer_line", boundedFollowupOfferLine ?: JSONObject.NULL)
            put("closure_style_tag", "LOCAL_RESULT_THEN_GENTLE_RETURN")
            put("ordered_houses", JSONArray().apply { orderedHouses.forEach { put(it) } })
            put("surviving_digits", JSONArray().apply { survivingDigits.forEach { put(it) } })
            put("asked_digit_survives", askedDigitSurvives)
            put("delay_reveal_until_resolution", storyArc.optBoolean("delay_reveal_until_resolution", false))
            put("must_not_open_with_merged_summary", storyArc.optBoolean("must_not_open_with_merged_summary", false))
            put("must_stage_house_pressure", storyArc.optBoolean("must_stage_house_pressure", false))
        }
    }

    fun buildFromNormalizedMoveProof(
        normalized: JSONObject,
        createdTurnSeq: Long
    ): DetourNarrativeContextV1? {
        if (normalized.optString("query_family", "").trim().uppercase() != "MOVE_PROOF") return null

        val anchor = normalized.optJSONObject("anchor") ?: JSONObject()
        val scope = normalized.optJSONObject("scope") ?: JSONObject()
        val question = normalized.optJSONObject("question") ?: JSONObject()
        val proofTruth = normalized.optJSONObject("proof_truth") ?: JSONObject()
        val answerTruth = normalized.optJSONObject("answer_truth") ?: JSONObject()
        val proofOutcome = proofTruth.optJSONObject("proof_outcome") ?: JSONObject()
        val methodFamily = firstNonBlank(
            normalized.optString("method_family", null),
            proofTruth.optString("method_family", null)
        )
        val answerPolarity = firstNonBlank(
            answerTruth.optString("answer_polarity", null),
            proofTruth.optString("answer_polarity", null)
        )

        val focusCells = stringList(scope.optJSONArray("cells"))
        val focusHouses = stringList(scope.optJSONArray("houses"))
        val overlayPolicy = overlayPolicyFromNormalized(normalized)
        val handbackPolicy = handbackPolicyFromNormalized(normalized)
        val narrativeArchetype = moveProofArchetypeFromNormalized(normalized, proofTruth)

        val survivorSummary = JSONObject().apply {
            val legacy = proofTruth.optJSONObject("survivor_summary")
            if (legacy != null) {
                val names = legacy.names()
                if (names != null) {
                    for (i in 0 until names.length()) {
                        val key = names.optString(i, "")
                        if (key.isNotBlank()) put(key, legacy.opt(key))
                    }
                }
            }
            if (proofOutcome.has("surviving_digit")) {
                put("surviving_digit", proofOutcome.opt("surviving_digit"))
            }
            if (proofOutcome.has("only_place_in_house")) {
                put("only_place_in_house", proofOutcome.opt("only_place_in_house"))
            }
            if (proofOutcome.has("winning_digit")) {
                put("winning_digit", proofOutcome.opt("winning_digit"))
            }
            if (proofOutcome.has("winning_cell")) {
                put("winning_cell", proofOutcome.opt("winning_cell"))
            }
            if (proofOutcome.has("nonproof_reason")) {
                put("nonproof_reason", proofOutcome.opt("nonproof_reason"))
            }
        }

        val contrastSummary = proofTruth.optJSONObject("contrast_summary") ?: JSONObject()
        val techniqueLegitimacy = proofTruth.optJSONObject("technique_legitimacy") ?: JSONObject()
        val proofLadder = proofTruth.optJSONObject("proof_ladder") ?: JSONObject()
        val storyFocus = proofTruth.optJSONObject("story_focus") ?: normalized.optJSONObject("story_focus") ?: JSONObject()
        val storyQuestion = proofTruth.optJSONObject("story_question") ?: normalized.optJSONObject("story_question") ?: JSONObject()
        val storyActors = proofTruth.optJSONObject("story_actors") ?: normalized.optJSONObject("story_actors") ?: JSONObject()
        val storyArc = proofTruth.optJSONObject("story_arc") ?: normalized.optJSONObject("story_arc") ?: JSONObject()
        val microStagePlan = proofTruth.optJSONObject("micro_stage_plan") ?: normalized.optJSONObject("micro_stage_plan") ?: JSONObject()
        val localProofGeometry = proofTruth.optJSONObject("local_proof_geometry") ?: normalized.optJSONObject("local_proof_geometry") ?: JSONObject()
        val closureContract = proofTruth.optJSONObject("closure_contract") ?: normalized.optJSONObject("closure_contract") ?: JSONObject()
        val visualLanguage = proofTruth.optJSONObject("visual_language") ?: normalized.optJSONObject("visual_language") ?: JSONObject()
        val evidenceSummary = moveProofEvidenceSummary(
            proofTruth = proofTruth,
            proofLadder = proofLadder
        )

        val commonFocusScope =
            firstNonBlank(
                storyFocus.optString("scope", null),
                scope.optString("kind", null),
                scope.optString("ref", null)
            )

        val claimKind = firstNonBlank(
            storyQuestion.optString("proof_object", null),
            proofTruth.optString("proof_object", null),
            proofTruth.optString("claim_kind", null),
            normalized.optString("proof_object", null),
            normalized.optString("query_profile", null)
        )

        val commonCentralQuestion = firstNonBlank(
            storyQuestion.optString("central_question", null),
            storyQuestion.optString("local_story_question", null),
            question.optString("central_question", null)
        )

        val commonAnswerBoundary = listOf(
            DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
            DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
            if ((proofTruth.optString("allowed_stage_boundary", "")).trim().uppercase() == "MAY_INCLUDE_COMMIT_FACT")
                DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
            else
                DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
        )

        val targetCell = firstNonBlank(
            question.optString("target_cell", null),
            storyActors.optString("target_cell", null)
        )
        val askedDigit = run {
            val q = question.optInt("asked_digit", -1)
            if (q in 1..9) {
                q
            } else {
                val s = storyActors.optInt("asked_digit", -1)
                if (s in 1..9) s else null
            }
        }
        val commonRivalCell = firstNonBlank(
            question.optString("rival_cell", null),
            contrastSummary.optString("rival_cell", null),
            storyActors.optString("rival_cell", null)
        )
        val commonClaimedTechniqueId = firstNonBlank(
            question.optString("claimed_technique_id", null),
            techniqueLegitimacy.optString("claimed_technique_id", null),
            storyActors.optJSONObject("technique_legitimacy")?.optString("claimed_technique_id", null)
        )

        val decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth)
        val permissibilitySupport =
            if (narrativeArchetype == DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN) {
                localPermissibilityNarrativeSupport(
                    storyArc = storyArc,
                    localProofGeometry = localProofGeometry,
                    targetCell = targetCell,
                    askedDigit = askedDigit,
                    focusHouses = focusHouses,
                    decisiveFact = decisiveFact,
                    handbackPolicy = handbackPolicy
                )
            } else {
                JSONObject()
            }

        val houseAlreadyOccupiedSupport =
            if (narrativeArchetype == DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED) {
                houseAlreadyOccupiedNarrativeSupport(
                    localProofGeometry = localProofGeometry,
                    focusHouses = focusHouses,
                    askedDigit = askedDigit,
                    decisiveFact = decisiveFact,
                    handbackPolicy = handbackPolicy
                )
            } else {
                JSONObject()
            }

        val filledCellSupport =
            if (narrativeArchetype == DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED) {
                filledCellNarrativeSupport(
                    localProofGeometry = localProofGeometry,
                    targetCell = targetCell,
                    askedDigit = askedDigit,
                    decisiveFact = decisiveFact,
                    handbackPolicy = handbackPolicy
                )
            } else {
                JSONObject()
            }

        val atom: DetourNarrativeAtomV1 =
            when (narrativeArchetype) {
                DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED ->
                    HouseAlreadyOccupiedAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        targetHouse = firstNonBlank(
                            houseAlreadyOccupiedSupport.optString("target_house", null),
                            focusHouses.firstOrNull()
                        ),
                        askedDigit = askedDigit,
                        existingDigitCell = firstNonBlank(
                            houseAlreadyOccupiedSupport.optString("existing_digit_cell", null),
                            targetCell
                        ),
                        openingFactLine = firstNonBlank(houseAlreadyOccupiedSupport.optString("opening_fact_line", null)),
                        duplicateRuleLine = firstNonBlank(houseAlreadyOccupiedSupport.optString("duplicate_rule_line", null)),
                        supportingSeatClosureLine = firstNonBlank(houseAlreadyOccupiedSupport.optString("supporting_seat_closure_line", null)),
                        boundedLandingLine = firstNonBlank(houseAlreadyOccupiedSupport.optString("bounded_landing_line", null)),
                        naturalReturnOfferLine = firstNonBlank(houseAlreadyOccupiedSupport.optString("natural_return_offer_line", null)),
                        boundedFollowupOfferLine = firstNonBlank(houseAlreadyOccupiedSupport.optString("bounded_followup_offer_line", null)),
                        closureStyleTag = firstNonBlank(houseAlreadyOccupiedSupport.optString("closure_style_tag", null)),
                        decisiveFact = decisiveFact,
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )

                DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED ->
                    FilledCellFactAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        targetCell = targetCell,
                        askedDigit = askedDigit,
                        placedValue = run {
                            val v = filledCellSupport.optInt("placed_value", -1)
                            if (v in 1..9) v else null
                        },
                        openingFactLine = firstNonBlank(filledCellSupport.optString("opening_fact_line", null)),
                        occupancyClarifierLine = firstNonBlank(filledCellSupport.optString("occupancy_clarifier_line", null)),
                        boundedLandingLine = firstNonBlank(filledCellSupport.optString("bounded_landing_line", null)),
                        naturalReturnOfferLine = firstNonBlank(filledCellSupport.optString("natural_return_offer_line", null)),
                        boundedFollowupOfferLine = firstNonBlank(filledCellSupport.optString("bounded_followup_offer_line", null)),
                        closureStyleTag = firstNonBlank(filledCellSupport.optString("closure_style_tag", null)),
                        decisiveFact = decisiveFact,
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )

                DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT ->
                    ContradictionSpotlightAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        targetCell = targetCell,
                        askedDigit = askedDigit,
                        blockerCell = run {
                            val rows = proofLadder.optJSONArray("rows") ?: JSONArray()
                            var found: String? = null
                            for (i in 0 until rows.length()) {
                                val row = rows.optJSONObject(i) ?: continue
                                val actorRef = firstNonBlank(row.optString("actor_ref", null))
                                if (!actorRef.isNullOrBlank() && actorRef != targetCell) {
                                    found = actorRef
                                    break
                                }
                            }
                            found ?: firstBlockerFromGeometry(localProofGeometry).first
                        },
                        blockerHouse = firstNonBlank(
                            proofTruth.optString("house_scope", null),
                            firstBlockerFromGeometry(localProofGeometry).second
                        ),
                        decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth),
                        evidenceSummary = evidenceSummary,

                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode

                    )

                DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN ->
                    LocalPermissibilityScanAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        targetCell = targetCell,
                        askedDigit = askedDigit,
                        openingSpotlightLine = firstNonBlank(permissibilitySupport.optString("opening_spotlight_line", null)),
                        openingSpotlightAlternates = permissibilitySupport.optJSONArray("opening_spotlight_alternates") ?: JSONArray(),
                        scanArenaLine = firstNonBlank(permissibilitySupport.optString("scan_arena_line", null)),
                        pressureBeats = permissibilitySupport.optJSONArray("pressure_beats") ?: JSONArray(),
                        survivorRevealLine = firstNonBlank(permissibilitySupport.optString("survivor_reveal_line", null)),
                        boundedLandingLine = firstNonBlank(permissibilitySupport.optString("bounded_landing_line", null)),
                        naturalReturnOfferLine = firstNonBlank(permissibilitySupport.optString("natural_return_offer_line", null)),
                        boundedFollowupOfferLine = firstNonBlank(permissibilitySupport.optString("bounded_followup_offer_line", null)),
                        closureStyleTag = firstNonBlank(permissibilitySupport.optString("closure_style_tag", null)),
                        orderedHouses = stringList(permissibilitySupport.optJSONArray("ordered_houses")),
                        survivingDigits = run {
                            val arr = permissibilitySupport.optJSONArray("surviving_digits") ?: JSONArray()
                            val out = mutableListOf<Int>()
                            for (i in 0 until arr.length()) {
                                val d = arr.optInt(i, -1)
                                if (d in 1..9) out += d
                            }
                            out.distinct().sorted()
                        },
                        askedDigitSurvives = permissibilitySupport.optBoolean("asked_digit_survives", false),
                        decisiveFact = decisiveFact,
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )

                DetourNarrativeArchetypeV1.SURVIVOR_LADDER ->
                    SurvivorLadderAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        targetCell = targetCell,
                        askedDigit = askedDigit,
                        ladderRows = proofLadder.optJSONArray("rows") ?: JSONArray(),
                        survivorSummary = survivorSummary,
                        decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth),

                        allowedStageBoundary = firstNonBlank(proofTruth.optString("allowed_stage_boundary", null)),
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode

                    )

                DetourNarrativeArchetypeV1.CONTRAST_DUEL ->
                    ContrastDuelAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        primaryCell = targetCell,
                        rivalCell = commonRivalCell,
                        askedDigit = askedDigit,
                        contrastSummary = contrastSummary,
                        ladderRows = proofLadder.optJSONArray("rows") ?: JSONArray(),
                        decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth),
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )

                DetourNarrativeArchetypeV1.PATTERN_LEGITIMACY_CHECK ->
                    PatternLegitimacyAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        centralQuestion = commonCentralQuestion,
                        claimedTechniqueId = commonClaimedTechniqueId,
                        legitimacySummary = techniqueLegitimacy,
                        ladderRows = proofLadder.optJSONArray("rows") ?: JSONArray(),
                        decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth),
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )

                DetourNarrativeArchetypeV1.HONEST_INSUFFICIENCY_ANSWER ->
                    HonestInsufficiencyAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        directAnswer = firstNonBlank(
                            answerTruth.optString("short_answer", null),
                            answerTruth.optString("one_sentence_claim", null)
                        ),
                        nonproofReason = firstNonBlank(proofOutcome.optString("nonproof_reason", null)),
                        localStateSummary = evidenceSummary,
                        decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth),
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )

                else ->
                    LocalProofSpotlightAtomV1(
                        focusScope = commonFocusScope,
                        focusCells = focusCells,
                        focusHouses = focusHouses,
                        claimKind = claimKind,
                        centralQuestion = commonCentralQuestion,
                        decisiveFact = moveProofDecisiveFact(answerTruth, proofTruth),
                        evidenceSummary = evidenceSummary,
                        survivorSummary = survivorSummary,
                        allowedStageBoundary = firstNonBlank(proofTruth.optString("allowed_stage_boundary", null)),
                        answerBoundary = commonAnswerBoundary,
                        handoverMode = handbackPolicy.handoverMode
                    )
            }

        val visualContract =
            buildMoveProofVisualContractV1(
                archetype = narrativeArchetype,
                atom = atom,
                overlayPolicy = overlayPolicy
            )

        val built = DetourNarrativeContextV1(
            demandCategory = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
            archetype = narrativeArchetype,
            dominantAtom = atom,
            anchorStepId = firstNonBlank(anchor.optString("step_id", null)),
            anchorStage = firstNonBlank(anchor.optString("story_stage", null)),
            pausedRouteCheckpointId = firstNonBlank(anchor.optString("paused_route_checkpoint_id", null)),
            overlayPolicy = overlayPolicy,
            handbackPolicy = handbackPolicy,
            visualContract = visualContract,

            narrativeSupport = JSONObject().apply {
                put("story_actors", storyActors)
                put("story_arc", storyArc)
                put("micro_stage_plan", microStagePlan)
                put("closure_contract", closureContract)
                put("visual_language", visualLanguage)

                if (narrativeArchetype == DetourNarrativeArchetypeV1.LOCAL_PERMISSIBILITY_SCAN) {
                    put("local_permissibility_support", permissibilitySupport)
                    put("preferred_opening_variants", permissibilitySupport.optJSONArray("opening_spotlight_alternates") ?: JSONArray())
                    put("preferred_return_offer_line", permissibilitySupport.optString("natural_return_offer_line", null) ?: JSONObject.NULL)
                    put("preferred_followup_offer_line", permissibilitySupport.optString("bounded_followup_offer_line", null) ?: JSONObject.NULL)
                    put("closure_style_tag", permissibilitySupport.optString("closure_style_tag", null) ?: JSONObject.NULL)
                }

                if (narrativeArchetype == DetourNarrativeArchetypeV1.HOUSE_ALREADY_OCCUPIED) {
                    put("house_already_occupied_support", houseAlreadyOccupiedSupport)
                    put("preferred_return_offer_line", houseAlreadyOccupiedSupport.optString("natural_return_offer_line", null) ?: JSONObject.NULL)
                    put("preferred_followup_offer_line", houseAlreadyOccupiedSupport.optString("bounded_followup_offer_line", null) ?: JSONObject.NULL)
                    put("closure_style_tag", houseAlreadyOccupiedSupport.optString("closure_style_tag", null) ?: JSONObject.NULL)
                }

                if (narrativeArchetype == DetourNarrativeArchetypeV1.CELL_ALREADY_FILLED) {
                    put("filled_cell_support", filledCellSupport)
                    put("preferred_return_offer_line", filledCellSupport.optString("natural_return_offer_line", null) ?: JSONObject.NULL)
                    put("preferred_followup_offer_line", filledCellSupport.optString("bounded_followup_offer_line", null) ?: JSONObject.NULL)
                    put("closure_style_tag", filledCellSupport.optString("closure_style_tag", null) ?: JSONObject.NULL)
                }
            },

            debugSupport = JSONObject().apply {
                put("local_proof_geometry", localProofGeometry)
                put("geometry_kind", firstNonBlank(localProofGeometry.optString("geometry_kind", null)) ?: JSONObject.NULL)
                put("proof_truth", proofTruth)
                put("answer_truth", answerTruth)
                put("proof_ladder", proofLadder)
            },

            createdTurnSeq = createdTurnSeq,
            sourceQueryFamily = "MOVE_PROOF",
            sourceQueryProfile = firstNonBlank(
                normalized.optString("query_profile", null),
                methodFamily
            )
        )

        runCatching {
            com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                tag = "DETOUR_PROOF_NATIVE_CONTEXT_V1",
                data = mapOf(
                    "created_turn_seq" to createdTurnSeq,
                    "query_family" to "MOVE_PROOF",
                    "query_profile" to (firstNonBlank(normalized.optString("query_profile", null), methodFamily) ?: "null"),
                    "challenge_lane" to (firstNonBlank(normalized.optString("challenge_lane", null), proofTruth.optString("challenge_lane", null)) ?: "null"),
                    "proof_object" to (firstNonBlank(normalized.optString("proof_object", null), proofTruth.optString("proof_object", null)) ?: "null"),
                    "method_family" to (methodFamily ?: "null"),
                    "geometry_kind" to (firstNonBlank(localProofGeometry.optString("geometry_kind", null)) ?: "null"),
                    "story_arc_opening_mode" to (firstNonBlank(storyArc.optString("opening_mode", null)) ?: "null"),
                    "story_arc_motion_mode" to (firstNonBlank(storyArc.optString("motion_mode", null)) ?: "null"),
                    "closure_mode" to (firstNonBlank(closureContract.optString("closure_mode", null)) ?: "null"),
                    "scan_language" to visualLanguage.optBoolean("may_use_scan_language", false),
                    "narrative_archetype" to built.archetype.name,
                    "dominant_atom_kind" to built.dominantAtom.atomKind,
                    "overlay_mode" to built.overlayPolicy.overlayMode.name,
                    "handover_mode" to built.handbackPolicy.handoverMode.name
                )
            )
        }

        return built
    }

    fun buildFromNormalizedLocalInspection(
        normalized: JSONObject,
        createdTurnSeq: Long
    ): DetourNarrativeContextV1? {
        if (normalized.optString("query_family", "").trim().uppercase() != "LOCAL_INSPECTION") return null

        val anchor = normalized.optJSONObject("anchor") ?: JSONObject()
        val scope = normalized.optJSONObject("scope") ?: JSONObject()
        val question = normalized.optJSONObject("question") ?: JSONObject()
        val inspectionTruth = normalized.optJSONObject("inspection_truth") ?: JSONObject()

        val focusCells = stringList(scope.optJSONArray("cells"))
        val focusHouses = stringList(scope.optJSONArray("houses"))
        val overlayPolicy = overlayPolicyFromNormalized(normalized)
        val handbackPolicy = handbackPolicyFromNormalized(normalized)

        val stateSummary = JSONArray().apply {
            val candidateState = inspectionTruth.optJSONObject("candidate_state")
            if (candidateState != null && candidateState.length() > 0) put(candidateState)
            val digitLocations = inspectionTruth.optJSONArray("digit_locations")
            if (digitLocations != null && digitLocations.length() > 0) put(JSONObject().apply {
                put("digit_locations", digitLocations)
            })
            val localDelta = inspectionTruth.optJSONObject("local_delta")
            if (localDelta != null && localDelta.length() > 0) put(localDelta)
            val nearby = inspectionTruth.optJSONObject("nearby_effects_summary")
            if (nearby != null && nearby.length() > 0) put(nearby)
            val constraints = inspectionTruth.optJSONObject("local_constraints_summary")
            if (constraints != null && constraints.length() > 0) put(constraints)
        }

        val atom = StateReadoutAtomV1(
            focusScope = firstNonBlank(scope.optString("kind", null), scope.optString("ref", null)),
            focusCells = focusCells,
            focusHouses = focusHouses,
            readoutKind = firstNonBlank(
                inspectionTruth.optString("state_read_kind", null),
                normalized.optString("query_profile", null)
            ),
            centralQuestion = firstNonBlank(question.optString("central_question", null)),
            stateSummary = stateSummary,
            whyItMatters = firstNonBlank(inspectionTruth.optString("why_it_matters", null)),
            answerBoundary = listOf(
                DetourAnswerBoundaryV1.DO_NOT_BECOME_PROOF_LADDER,
                DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
            ),
            handoverMode = handbackPolicy.handoverMode
        )

        return DetourNarrativeContextV1(
            demandCategory = DetourDemandCategoryV2.LOCAL_GRID_INSPECTION,
            archetype = DetourNarrativeArchetypeV1.STATE_READOUT,
            dominantAtom = atom,
            anchorStepId = firstNonBlank(anchor.optString("step_id", null)),
            anchorStage = firstNonBlank(anchor.optString("story_stage", null)),
            pausedRouteCheckpointId = firstNonBlank(anchor.optString("paused_route_checkpoint_id", null)),
            overlayPolicy = overlayPolicy,
            handbackPolicy = handbackPolicy,
            visualContract = buildLocalInspectionVisualContractV1(
                atom = atom,
                overlayPolicy = overlayPolicy
            ),
            createdTurnSeq = createdTurnSeq,
            sourceQueryFamily = "LOCAL_INSPECTION",
            sourceQueryProfile = firstNonBlank(normalized.optString("query_profile", null))
        )
    }

    fun buildFromNormalizedProposalVerdict(
        normalized: JSONObject,
        createdTurnSeq: Long
    ): DetourNarrativeContextV1? {
        if (normalized.optString("query_family", "").trim().uppercase() != "PROPOSAL_VERDICT") return null

        val anchor = normalized.optJSONObject("anchor") ?: JSONObject()
        val scope = normalized.optJSONObject("scope") ?: JSONObject()
        val question = normalized.optJSONObject("question") ?: JSONObject()
        val proposalTruth = normalized.optJSONObject("proposal_truth") ?: JSONObject()

        val focusCells = stringList(scope.optJSONArray("cells"))
        val focusHouses = stringList(scope.optJSONArray("houses"))
        val overlayPolicy = overlayPolicyFromNormalized(normalized)
        val handbackPolicy = handbackPolicyFromNormalized(normalized)

        val atom = ProposalVerdictAtomV1(
            focusScope = firstNonBlank(scope.optString("kind", null), scope.optString("ref", null)),
            focusCells = focusCells,
            focusHouses = focusHouses,
            proposalSummary = firstNonBlank(
                question.optString("proposal_summary", null),
                question.optString("proposal_text", null)
            ),
            verdict = firstNonBlank(proposalTruth.optString("verdict", null)),
            whatIsCorrect = copyArray(proposalTruth.optJSONArray("what_is_correct")),
            whatIsIncorrect = copyArray(proposalTruth.optJSONArray("what_is_incorrect")),
            missingCondition = firstNonBlank(proposalTruth.optString("missing_condition", null)),
            routeRelation = firstNonBlank(proposalTruth.optString("route_alignment", null)),
            evidenceSummary = copyArray(proposalTruth.optJSONArray("support_rows")),
            answerBoundary = listOf(
                DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
                DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
            ),
            handoverMode = handbackPolicy.handoverMode
        )

        return DetourNarrativeContextV1(
            demandCategory = DetourDemandCategoryV2.USER_PROPOSAL_VERDICT,
            archetype = DetourNarrativeArchetypeV1.PROPOSAL_VERDICT,
            dominantAtom = atom,
            anchorStepId = firstNonBlank(anchor.optString("step_id", null)),
            anchorStage = firstNonBlank(anchor.optString("story_stage", null)),
            pausedRouteCheckpointId = firstNonBlank(anchor.optString("paused_route_checkpoint_id", null)),
            overlayPolicy = overlayPolicy,
            handbackPolicy = handbackPolicy,
            visualContract = buildProposalVerdictVisualContractV1(
                atom = atom,
                overlayPolicy = overlayPolicy
            ),
            createdTurnSeq = createdTurnSeq,
            sourceQueryFamily = "PROPOSAL_VERDICT",
            sourceQueryProfile = firstNonBlank(normalized.optString("query_profile", null))
        )
    }
}