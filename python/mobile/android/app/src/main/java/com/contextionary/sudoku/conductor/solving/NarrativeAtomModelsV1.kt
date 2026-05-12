    package com.contextionary.sudoku.conductor.solving

    import org.json.JSONArray
    import org.json.JSONObject

    /**
     * NarrativeAtomModelsV1
     *
     * Full replacement aligned to the new normalize_step.py contract.
     *
     * Key architectural changes:
     * - Reads canonical proof.applications[*] emitted by the new Python normalizer.
     * - Stops guessing archetype from technique names when narrative.archetype is available.
     * - Preserves rich Singles behavior via support.explanation_links.
     * - Supports non-singles from canonical pattern/effects sections.
     * - Keeps the existing deterministic atom ladder style:
     *      spotlight -> witness/setup -> lock-in -> commit
     * - Validator is now claim-aware / archetype-aware instead of singles-only.
     *
     * Wave-0 family constitution notes:
     * - SUBSETS = advanced pattern-emergence family
     * - INTERSECTIONS = advanced territorial-control family
     *
     * INTERSECTIONS North Star:
     * - Setup opens at the overlap / crossroads between two houses.
     * - Setup explicitly audits the source house outside the overlap before naming
     *   the pattern.
     * - Confrontation uses a two-actor structure:
     *      ordinary witnesses first, territorial-control hero second.
     * - Resolution distinguishes indirect pattern power from direct final survivor.
     *
     * This file does not implement the new Wave-1+ intersection pipeline yet; the
     * purpose of these notes is to pin the intended family identity before behavior
     * patches land.
     */

    // ----------------------------
    // Compact enums (wire-level strings)
    // ----------------------------

    enum class NarrativeArchetypeV1(val wire: String) {
        HIDDEN_SINGLES("HIDDEN_SINGLES"),
        NAKED_SINGLES("NAKED_SINGLES"),
        FULL_HOUSE("FULL_HOUSE"),

        // Advanced family: local pattern emergence (naked/hidden pair/triple/quad...)
        SUBSETS("SUBSETS"),

        // Advanced family: territorial control at a box-line overlap
        // (claiming pair/triple + pointing pair/triple).
        INTERSECTIONS("INTERSECTIONS"),

        FISH("FISH"),
        WINGS("WINGS"),
        CHAINS("CHAINS"),
        UNKNOWN("UNKNOWN");

        companion object {
            fun fromWire(raw: String?): NarrativeArchetypeV1 =
                entries.firstOrNull { it.wire.equals(raw.orEmpty(), ignoreCase = true) } ?: UNKNOWN
        }
    }

    enum class NarrativeBeatKindV1(val wire: String) {
        SPOTLIGHT("SPOTLIGHT"),
        WITNESS_ELIMINATION("WITNESS_ELIMINATION"),
        LOCK_IN("LOCK_IN"),
        COMMIT("COMMIT"),
        AFTERGLOW("AFTERGLOW"),
        TEACHING_NOTE("TEACHING_NOTE")
    }

    enum class SpoilerLevelV1(val wire: String) {
        NONE("NONE"),
        CANDIDATE("CANDIDATE"),
        DIGIT("DIGIT")
    }

    enum class NarrativeClaimCodeV1(val wire: String) {
        SEARCH_DIGIT_IN_HOUSE("SEARCH_DIGIT_IN_HOUSE"),
        SEARCH_DIGITS_IN_CELL("SEARCH_DIGITS_IN_CELL"),
        CELL_CANNOT_BE_DIGIT("CELL_CANNOT_BE_DIGIT"),
        ONLY_CELL_LEFT_FOR_DIGIT_IN_HOUSE("ONLY_CELL_LEFT_FOR_DIGIT_IN_HOUSE"),
        ONLY_DIGIT_LEFT_FOR_CELL("ONLY_DIGIT_LEFT_FOR_CELL"),
        PLACE_DIGIT("PLACE_DIGIT"),

        SUBSET_LOCKS_DIGITS("SUBSET_LOCKS_DIGITS"),
        DIGIT_LOCKED_TO_LINE_IN_BOX("DIGIT_LOCKED_TO_LINE_IN_BOX"),
        FISH_LOCKS_DIGIT("FISH_LOCKS_DIGIT"),
        EITHER_WAY_ELIMINATION("EITHER_WAY_ELIMINATION"),
        CONTRADICTION_IMPLES_NOT("CONTRADICTION_IMPLES_NOT"),

        OTHER("OTHER")
    }

    enum class NarrativeWitnessKindV1(val wire: String) {
        BLOCKS_CANDIDATE("BLOCKS_CANDIDATE"),
        HOUSE_REMAINS_ONE("HOUSE_REMAINS_ONE"),
        CELL_REMAINS_ONE_DIGIT("CELL_REMAINS_ONE_DIGIT"),

        SUBSET_DEFINITION("SUBSET_DEFINITION"),
        INTERSECTION_LOCK("INTERSECTION_LOCK"),
        CONJUGATE_PAIR("CONJUGATE_PAIR"),
        SEES_BOTH("SEES_BOTH"),
        CHAIN_CONTRADICTION("CHAIN_CONTRADICTION"),

        OTHER("OTHER")
    }

    enum class OverlayRoleV1(val wire: String) {
        focus("focus"),
        peer("peer"),
        witness("witness"),
        eliminate_digit("eliminate_digit"),
        lock_digit("lock_digit"),
        link_a("link_a"),
        link_b("link_b"),
        contradiction("contradiction"),
        result_place("result_place"),
        primary_house("primary_house"),
        secondary_house("secondary_house"),
        sweep_house("sweep_house")
    }

    enum class OverlayIntentV1(val wire: String) {
        SHOW_SPOTLIGHT("SHOW_SPOTLIGHT"),
        SHOW_WITNESS("SHOW_WITNESS"),
        SHOW_SWEEP("SHOW_SWEEP"),
        SHOW_COMMIT("SHOW_COMMIT")
    }

    // ----------------------------
    // Generator
    // ----------------------------

    object NarrativeAtomGeneratorV1 {

        fun buildNarrativeAtomsV1(stepObj: JSONObject, grid81: String?): JSONObject {
            val grid = (grid81 ?: "").takeIf { it.length == 81 } ?: return emptyPacket("grid_missing")

            val targetObj = stepObj.optJSONObject("target") ?: JSONObject()
            val targetCell = targetObj.optJSONObject("cell") ?: JSONObject()
            val tr = targetCell.optInt("r", -1)
            val tc = targetCell.optInt("c", -1)
            val tIdx = targetCell.optInt("cellIndex", targetCell.optInt("cell_index", -1))
            val td = targetObj.optInt("digit", -1)

            if (tr !in 1..9 || tc !in 1..9 || tIdx !in 0..80 || td !in 1..9) {
                return emptyPacket("target_missing")
            }

            val leadApp = StepSemanticReaderV2.readLeadApplication(stepObj)
            val archetype = StepSemanticReaderV2.readArchetype(stepObj, leadApp)
            val primaryHouse = StepSemanticReaderV2.pickPrimaryHouse(stepObj, leadApp, tr, tc, tIdx)

            // Truth-first reroutes:
            // - SUBSETS render from narrative_truth_v2
            // - INTERSECTIONS now also render from narrative_truth_v2
            if (archetype == NarrativeArchetypeV1.SUBSETS) {
                return buildSubsetAtomsPacketFromTruthV2(
                    stepObj = stepObj,
                    grid81 = grid,
                    tr = tr,
                    tc = tc,
                    tIdx = tIdx,
                    td = td,
                    fallbackPrimaryHouse = primaryHouse
                )
            }

            if (archetype == NarrativeArchetypeV1.INTERSECTIONS) {
                return buildIntersectionAtomsPacketFromTruthV2(
                    stepObj = stepObj,
                    grid81 = grid,
                    tr = tr,
                    tc = tc,
                    tIdx = tIdx,
                    td = td,
                    fallbackPrimaryHouse = primaryHouse
                )
            }

            val atoms = JSONArray()

            atoms.put(
                atomSpotlight(
                    index = 0,
                    archetype = archetype,
                    tr = tr,
                    tc = tc,
                    tIdx = tIdx,
                    td = td,
                    primaryHouse = primaryHouse
                )
            )

            when (archetype) {
                NarrativeArchetypeV1.HIDDEN_SINGLES -> {
                    val hs = StepSemanticReaderV2.extractCanonicalHiddenSingle(stepObj, leadApp, tIdx, td)
                        ?: return emptyPacket("hidden_single_canonical_missing")

                    var nextIndex = 1
                    val maxWitnessAtoms = 8

                    for ((peer, witness) in hs.peerWitnessPairs.take(maxWitnessAtoms)) {
                        atoms.put(
                            atomWitnessElimination(
                                index = nextIndex++,
                                archetype = archetype,
                                focusIdx = hs.focusIdx,
                                focusR = hs.focusR,
                                focusC = hs.focusC,
                                digit = hs.digit,
                                primaryHouse = hs.primaryHouse,
                                peerIdx = peer.cellIndex,
                                witnessIdx = witness.cellIndex,
                                witnessDigit = hs.digit,
                                relation = relationBetweenCells(peer.cellIndex, witness.cellIndex)
                            )
                        )
                    }

                    atoms.put(
                        atomLockIn(
                            index = nextIndex++,
                            archetype = archetype,
                            tr = hs.focusR,
                            tc = hs.focusC,
                            tIdx = hs.focusIdx,
                            td = hs.digit,
                            primaryHouse = hs.primaryHouse,
                            eliminatedPeers = hs.peerWitnessPairs.map { it.first.cellIndex },
                            eliminatedDigits = emptyList()
                        )
                    )

                    atoms.put(
                        atomCommit(
                            index = nextIndex,
                            archetype = archetype,
                            tr = hs.focusR,
                            tc = hs.focusC,
                            tIdx = hs.focusIdx,
                            td = hs.digit,
                            primaryHouse = hs.primaryHouse
                        )
                    )
                }

                NarrativeArchetypeV1.NAKED_SINGLES -> {
                    val ns = StepSemanticReaderV2.extractCanonicalNakedSingle(stepObj, leadApp, tIdx, td)
                        ?: return emptyPacket("naked_single_canonical_missing")

                    var nextIndex = 1
                    val maxWitnessAtoms = 8

                    for ((elimDigit, witness) in ns.digitWitnessPairs.take(maxWitnessAtoms)) {
                        atoms.put(
                            atomWitnessElimination(
                                index = nextIndex++,
                                archetype = archetype,
                                focusIdx = ns.focusIdx,
                                focusR = ns.focusR,
                                focusC = ns.focusC,
                                digit = elimDigit,
                                primaryHouse = ns.primaryHouse,
                                peerIdx = ns.focusIdx,
                                witnessIdx = witness.cellIndex,
                                witnessDigit = elimDigit,
                                relation = relationBetweenCells(ns.focusIdx, witness.cellIndex)
                            )
                        )
                    }

                    atoms.put(
                        atomLockIn(
                            index = nextIndex++,
                            archetype = archetype,
                            tr = ns.focusR,
                            tc = ns.focusC,
                            tIdx = ns.focusIdx,
                            td = ns.digit,
                            primaryHouse = ns.primaryHouse,
                            eliminatedPeers = emptyList(),
                            eliminatedDigits = ns.eliminatedDigits
                        )
                    )

                    atoms.put(
                        atomCommit(
                            index = nextIndex,
                            archetype = archetype,
                            tr = ns.focusR,
                            tc = ns.focusC,
                            tIdx = ns.focusIdx,
                            td = ns.digit,
                            primaryHouse = ns.primaryHouse
                        )
                    )
                }

                NarrativeArchetypeV1.FULL_HOUSE -> {
                    val fh = StepSemanticReaderV2.extractCanonicalFullHouse(
                        stepObj = stepObj,
                        leadApp = leadApp,
                        grid81 = grid,
                        fallbackTargetIdx = tIdx,
                        fallbackDigit = td
                    ) ?: return emptyPacket("full_house_canonical_missing")

                    atoms.put(
                        atomLockIn(
                            index = 1,
                            archetype = archetype,
                            tr = fh.focusR,
                            tc = fh.focusC,
                            tIdx = fh.focusIdx,
                            td = fh.digit,
                            primaryHouse = fh.primaryHouse,
                            eliminatedPeers = emptyList(),
                            eliminatedDigits = emptyList()
                        )
                    )

                    atoms.put(
                        atomCommit(
                            index = 2,
                            archetype = archetype,
                            tr = fh.focusR,
                            tc = fh.focusC,
                            tIdx = fh.focusIdx,
                            td = fh.digit,
                            primaryHouse = fh.primaryHouse
                        )
                    )
                }

                NarrativeArchetypeV1.INTERSECTIONS -> {
                    // Handled earlier through buildIntersectionAtomsPacketFromTruthV2(...)
                    return emptyPacket("intersection_reroute_failed")
                }

                NarrativeArchetypeV1.SUBSETS -> {
                    // Handled earlier through buildSubsetAtomsPacketFromTruthV2(...)
                    return emptyPacket("subset_reroute_failed")
                }

                NarrativeArchetypeV1.FISH -> {
                    val fish = StepSemanticReaderV2.extractFishData(stepObj, leadApp, td)
                    var nextIndex = 1

                    if (fish != null) {
                        atoms.put(atomFishPattern(nextIndex++, fish))
                        atoms.put(atomFishSweep(nextIndex++, fish))
                    } else {
                        atoms.put(atomTeachingNoteFish(nextIndex++, tr, tc, tIdx, td, primaryHouse))
                    }

                    atoms.put(
                        atomLockIn(
                            index = nextIndex++,
                            archetype = archetype,
                            tr = tr,
                            tc = tc,
                            tIdx = tIdx,
                            td = td,
                            primaryHouse = primaryHouse,
                            eliminatedPeers = emptyList(),
                            eliminatedDigits = emptyList()
                        )
                    )

                    atoms.put(atomCommit(nextIndex, archetype, tr, tc, tIdx, td, primaryHouse))
                }

                NarrativeArchetypeV1.WINGS -> {
                    val wing = StepSemanticReaderV2.extractWingData(stepObj, leadApp, td)
                    var nextIndex = 1

                    if (wing != null) {
                        atoms.put(atomWingEitherWay(nextIndex++, wing))
                        atoms.put(atomWingElimination(nextIndex++, wing))
                    } else {
                        atoms.put(atomTeachingNoteWing(nextIndex++, tr, tc, tIdx, td, primaryHouse))
                    }

                    atoms.put(
                        atomLockIn(
                            index = nextIndex++,
                            archetype = archetype,
                            tr = tr,
                            tc = tc,
                            tIdx = tIdx,
                            td = td,
                            primaryHouse = primaryHouse,
                            eliminatedPeers = emptyList(),
                            eliminatedDigits = emptyList()
                        )
                    )

                    atoms.put(atomCommit(nextIndex, archetype, tr, tc, tIdx, td, primaryHouse))
                }

                NarrativeArchetypeV1.CHAINS -> {
                    val chain = StepSemanticReaderV2.extractChainData(stepObj, leadApp, td)
                    var nextIndex = 1

                    if (chain != null) {
                        atoms.put(atomChainColoring(nextIndex++, chain))
                        atoms.put(atomChainContradiction(nextIndex++, chain))
                    } else {
                        atoms.put(atomTeachingNoteChain(nextIndex++, tr, tc, tIdx, td, primaryHouse))
                    }

                    atoms.put(
                        atomLockIn(
                            index = nextIndex++,
                            archetype = archetype,
                            tr = tr,
                            tc = tc,
                            tIdx = tIdx,
                            td = td,
                            primaryHouse = primaryHouse,
                            eliminatedPeers = emptyList(),
                            eliminatedDigits = emptyList()
                        )
                    )

                    atoms.put(atomCommit(nextIndex, archetype, tr, tc, tIdx, td, primaryHouse))
                }

                NarrativeArchetypeV1.UNKNOWN -> {
                    atoms.put(
                        atomLockIn(
                            index = 1,
                            archetype = archetype,
                            tr = tr,
                            tc = tc,
                            tIdx = tIdx,
                            td = td,
                            primaryHouse = primaryHouse,
                            eliminatedPeers = emptyList(),
                            eliminatedDigits = emptyList()
                        )
                    )
                    atoms.put(atomCommit(2, archetype, tr, tc, tIdx, td, primaryHouse))
                }
            }

            val truth = buildNarrativeTruthV2(stepObj, grid81)
            val finalResolution =
                truth.optJSONObject("final_resolution")?.let { JSONObject(it.toString()) } ?: JSONObject()

            val packet = JSONObject().apply {
                put("schema_version", "narrative_packet_v1")
                put("evidence", JSONObject().apply {
                    put("narrative_truth_v2", JSONObject(truth.toString()))
                    put("narrative_atoms_v1", JSONObject().apply {
                        put("schema_version", "narrative_atoms_v1")
                        put("archetype", archetype.wire)
                        put("atoms", JSONArray(atoms.toString()))
                        put("final_resolution", JSONObject(finalResolution.toString()))
                        put("version_note", "canonical_reader_v2_final_resolution_aligned_stage_scope_validated")
                    })
                })
            }

            val problems = JSONArray()
            ProofValidatorV1.validateFinalResolutionContract(
                packet = packet,
                atoms = atoms,
                finalResolution = finalResolution,
                problems = problems
            )

            val audit = JSONObject().apply {
                put("kind", "NARRATIVE_PACKET_AUDIT_V2")
                put("atom0_snapshot", ProofValidatorV1.buildAtom0AuditSnapshotV2(packet))
                put("summary", JSONObject().apply {
                    put("validation_problem_count", problems.length())
                    put("validation_status", if (problems.length() > 0) "invalid" else "ok")
                })
            }

            return JSONObject().apply {
                put("schema_version", "narrative_atoms_v1")
                put("archetype", archetype.wire)
                put("atoms", atoms)
                put("final_resolution", finalResolution)
                put("audit", audit)
                put(
                    "validation",
                    JSONObject().apply {
                        put("status", if (problems.length() > 0) "invalid" else "ok")
                        put("problems", problems)
                    }
                )
                put("version_note", "canonical_reader_v2_final_resolution_aligned_stage_scope_validated_trigger_packet_audited")
            }
        }

        fun buildNarrativeTruthV2(stepObj: JSONObject, grid81: String?): JSONObject {
            val leadApp = StepSemanticReaderV2.readLeadApplication(stepObj)
            val archetype = StepSemanticReaderV2.readArchetype(stepObj, leadApp)

            val techObj = stepObj.optJSONObject("technique") ?: JSONObject()
            val proofObj = stepObj.optJSONObject("proof") ?: JSONObject()
            val targetObj = stepObj.optJSONObject("target") ?: JSONObject()

            val placement = firstPlacement(proofObj.optJSONArray("placements"))
            val targetCell = parseCellRef(targetObj.opt("cell")) ?: placement?.cell
            val targetDigit = targetObj.optInt("digit", placement?.digit ?: -1).takeIf { it in 1..9 }

            val techniqueId = techObj.optString("technique_id")
                .ifBlank { techObj.optString("id") }
                .ifBlank { techObj.optString("app_name") }

            val techniqueName = techObj.optString("technique_name")
                .ifBlank { techObj.optString("real_name") }
                .ifBlank { techObj.optString("name") }
                .ifBlank { techniqueId }

            val family = techObj.optString("family")
            val isBase = techObj.optBoolean("is_base", false)

            val out = JSONObject().apply {
                put("schema_version", "narrative_truth_v2")
                put("source_of_truth", "packet.evidence.narrative_truth_v2")

                put("technique", JSONObject().apply {
                    put("technique_id", techniqueId)
                    put("technique_name", techniqueName)
                    put("family", family)
                    put("archetype", archetype.wire)
                    put("is_base", isBase)
                })

                put("focus", JSONObject().apply {
                    if (targetDigit != null) put("digit", targetDigit)
                    if (targetCell != null) {
                        put("focus_cell", targetCell.toJson())
                        put("focus_cell_label", "r${targetCell.r}c${targetCell.c} (${targetCell.cellIndex})")
                    }
                })

                put("proof_payload", JSONObject())
                put("witness_pattern", JSONObject())

                // Phase 2 / 7A: advanced atom-0 truth should be explicit and reusable.
                put("trigger_pattern", JSONObject())
                put("trigger_explanation", JSONObject())
                put("trigger_bridge", JSONObject())
                put("trigger_packet", JSONObject())

                put("downstream_resolution", JSONObject())

                put("final_resolution", JSONObject())
                put("traceability", JSONObject().apply {
                    put("derived_from", JSONObject().apply {
                        put(
                            "lead_application_id",
                            leadApp?.optJSONObject("identity")?.optString("application_id")
                                ?.takeIf { it.isNotBlank() } ?: JSONObject.NULL
                        )
                        put(
                            "lead_pattern_subtype",
                            leadApp?.optJSONObject("pattern")?.optString("pattern_subtype")
                                ?.takeIf { it.isNotBlank() } ?: JSONObject.NULL
                        )
                        put(
                            "step_target_digit",
                            targetDigit ?: JSONObject.NULL
                        )
                    })
                })
            }

            when (archetype) {
                NarrativeArchetypeV1.HIDDEN_SINGLES -> {
                    if (targetCell != null && targetDigit != null) {
                        val hs = StepSemanticReaderV2.extractCanonicalHiddenSingle(stepObj, leadApp, targetCell.cellIndex, targetDigit)
                        if (hs != null) {
                            out.put("resolution_kind", "HOUSE_CANDIDATE_CELLS_FOR_DIGIT")
                            out.put("primary_house", JSONObject(hs.primaryHouse.toString()))
                            out.put("proof_payload", buildHiddenSingleProofPayloadV2(hs))
                            out.put("witness_pattern", JSONObject().apply {
                                put("kind", "BASE_HIDDEN_SINGLE")
                            })
                            out.put("downstream_resolution", buildDownstreamResolutionV2(targetCell, targetDigit))
                            out.put(
                                "final_resolution",
                                buildFinalResolutionContractV2(
                                    kind = "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
                                    primaryHouse = hs.primaryHouse,
                                    focusCell = targetCell,
                                    digit = targetDigit,
                                    originStory = JSONObject().apply {
                                        put("kind", "BASE_HIDDEN_SINGLE")
                                    }
                                )
                            )
                        }
                    }
                }

                NarrativeArchetypeV1.NAKED_SINGLES -> {
                    if (targetCell != null && targetDigit != null) {
                        val ns = StepSemanticReaderV2.extractCanonicalNakedSingle(stepObj, leadApp, targetCell.cellIndex, targetDigit)
                        if (ns != null) {
                            out.put("resolution_kind", "CELL_CANDIDATE_DIGITS")
                            out.put("primary_house", cellHouseRef(ns.focusIdx))
                            out.put("proof_payload", buildNakedSingleProofPayloadV2(ns))
                            out.put("witness_pattern", JSONObject().apply {
                                put("kind", "BASE_NAKED_SINGLE")
                            })
                            out.put("downstream_resolution", buildDownstreamResolutionV2(targetCell, targetDigit))
                            out.put(
                                "final_resolution",
                                buildFinalResolutionContractV2(
                                    kind = "CELL_CANDIDATE_DIGITS",
                                    primaryHouse = cellHouseRef(ns.focusIdx),
                                    focusCell = CellRef(ns.focusR, ns.focusC, ns.focusIdx),
                                    digit = ns.digit,
                                    originStory = JSONObject().apply {
                                        put("kind", "BASE_NAKED_SINGLE")
                                    }
                                )
                            )
                        }
                    }
                }

                NarrativeArchetypeV1.FULL_HOUSE -> {
                    if (targetCell != null && targetDigit != null) {
                        val fh = StepSemanticReaderV2.extractCanonicalFullHouse(
                            stepObj = stepObj,
                            leadApp = leadApp,
                            grid81 = grid81,
                            fallbackTargetIdx = targetCell.cellIndex,
                            fallbackDigit = targetDigit
                        )
                        if (fh != null) {
                            out.put("resolution_kind", "HOUSE_CANDIDATE_CELLS_FOR_DIGIT")
                            out.put("primary_house", JSONObject(fh.primaryHouse.toString()))
                            out.put("proof_payload", buildFullHouseProofPayloadV2(fh))
                            out.put("witness_pattern", JSONObject().apply {
                                put("kind", "BASE_FULL_HOUSE")
                            })
                            out.put("downstream_resolution", buildDownstreamResolutionV2(targetCell, targetDigit))
                            out.put(
                                "final_resolution",
                                buildFinalResolutionContractV2(
                                    kind = "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
                                    primaryHouse = fh.primaryHouse,
                                    focusCell = CellRef(fh.focusR, fh.focusC, fh.focusIdx),
                                    digit = fh.digit,
                                    originStory = JSONObject().apply {
                                        put("kind", "BASE_FULL_HOUSE")
                                    }
                                )
                            )
                        }
                    }
                }

                NarrativeArchetypeV1.INTERSECTIONS -> {
                    val canonical = StepSemanticReaderV2.extractCanonicalIntersection(
                        stepObj = stepObj,
                        leadApp = leadApp,
                        fallbackTargetCell = targetCell,
                        fallbackTargetDigit = targetDigit
                    )

                    if (canonical != null) {
                        out.put("resolution_kind", "INTERSECTION_SWEEP")
                        out.put("primary_house", JSONObject(canonical.primaryHouse.toString()))

                        val proofPayload = buildIntersectionProofPayloadV2(canonical)
                        val triggerPacket = buildIntersectionTriggerPacketV2(canonical, proofPayload)
                        val triggerPattern = triggerPacket.optJSONObject("trigger_pattern") ?: JSONObject()
                        val triggerExplanation = triggerPacket.optJSONObject("trigger_explanation") ?: JSONObject()
                        val triggerBridge = triggerPacket.optJSONObject("trigger_bridge") ?: JSONObject()

                        out.put("proof_payload", proofPayload)
                        out.put("witness_pattern", JSONObject(triggerPattern.toString()))
                        out.put("trigger_pattern", triggerPattern)
                        out.put("trigger_explanation", triggerExplanation)
                        out.put("trigger_bridge", triggerBridge)
                        out.put("trigger_packet", triggerPacket)

                        val downstream = buildIntersectionDownstreamResolutionV2(canonical)
                        if (downstream.length() > 0) {
                            out.put("downstream_resolution", downstream)
                        }

                        out.put(
                            "final_resolution",
                            buildFinalResolutionContractV2(
                                kind = canonical.finalResolutionKind,
                                primaryHouse = canonical.finalPrimaryHouse,
                                focusCell = canonical.finalFocusCell,
                                digit = canonical.finalTargetDigit,
                                originStory = JSONObject().apply {
                                    put("kind", "INTERSECTION_ORIGIN")
                                    put("interaction_kind", canonical.interactionKind)
                                    put("source_house", JSONObject(canonical.sourceHouse.toString()))
                                    put("target_house", JSONObject(canonical.targetHouse.toString()))
                                    put(
                                        "source_confinement_proof",
                                        canonical.sourceConfinementProof?.let { JSONObject(it.toString()) } ?: JSONObject.NULL
                                    )
                                }
                            )
                        )
                    }
                }

                NarrativeArchetypeV1.SUBSETS -> {
                    val fallbackHouse = targetCell?.let { cellHouseRef(it.cellIndex) } ?: JSONObject().apply {
                        put("type", "cell")
                    }

                    val canonical = StepSemanticReaderV2.extractCanonicalSubset(
                        stepObj = stepObj,
                        leadApp = leadApp,
                        fallbackPrimaryHouse = fallbackHouse,
                        fallbackTargetCell = targetCell,
                        fallbackTargetDigit = targetDigit
                    )

                    if (canonical != null) {
                        out.put("resolution_kind", canonical.eliminationKind)
                        out.put("primary_house", JSONObject(canonical.primaryHouse.toString()))

                        val proofPayload = buildSubsetProofPayloadV2(canonical)
                        val triggerPacket = buildSubsetTriggerPacketV2(stepObj, grid81, canonical, proofPayload)
                        val triggerPattern = triggerPacket.optJSONObject("trigger_pattern") ?: JSONObject()
                        val triggerExplanation = triggerPacket.optJSONObject("trigger_explanation") ?: JSONObject()
                        val triggerBridge = triggerPacket.optJSONObject("trigger_bridge") ?: JSONObject()

                        out.put("proof_payload", proofPayload)
                        out.put("witness_pattern", JSONObject(triggerPattern.toString()))
                        out.put("trigger_pattern", triggerPattern)
                        out.put("trigger_explanation", triggerExplanation)
                        out.put("trigger_bridge", triggerBridge)
                        out.put("trigger_packet", triggerPacket)

                        val downstreamCell = canonical.focusCell ?: targetCell
                        val downstreamDigit = canonical.remainingCandidateDigits.firstOrNull()
                            ?: canonical.targetDigit
                            ?: targetDigit

                        if (downstreamCell != null && downstreamDigit != null && downstreamDigit in 1..9) {
                            out.put("downstream_resolution", buildDownstreamResolutionV2(downstreamCell, downstreamDigit))
                        }

                        out.put(
                            "final_resolution",
                            buildFinalResolutionContractV2(
                                kind = canonical.eliminationKind,
                                primaryHouse = canonical.primaryHouse,
                                focusCell = canonical.focusCell ?: targetCell,
                                digit = downstreamDigit,
                                originStory = JSONObject().apply {
                                    put("kind", "SUBSET_ORIGIN")
                                    put("subset_mode", canonical.subset.subsetMode)
                                    put("subset_subtype", canonical.subset.subsetSubtype)
                                    put("house", JSONObject(canonical.subset.house.toString()))
                                }
                            )
                        )
                    }
                }

                else -> {
                    if (targetCell != null && targetDigit != null) {
                        out.put("downstream_resolution", buildDownstreamResolutionV2(targetCell, targetDigit))
                        out.put(
                            "final_resolution",
                            buildFinalResolutionContractV2(
                                kind = "CELL_CANDIDATE_DIGITS",
                                primaryHouse = cellHouseRef(targetCell.cellIndex),
                                focusCell = targetCell,
                                digit = targetDigit,
                                originStory = JSONObject().apply {
                                    put("kind", "UNKNOWN")
                                }
                            )
                        )
                    }
                    out.put("witness_pattern", JSONObject().apply {
                        put(
                            "kind",
                            when (archetype) {
                                NarrativeArchetypeV1.INTERSECTIONS -> "INTERSECTION"
                                NarrativeArchetypeV1.FISH -> "FISH"
                                NarrativeArchetypeV1.WINGS -> "WING"
                                NarrativeArchetypeV1.CHAINS -> "CHAIN"
                                else -> "UNKNOWN"
                            }
                        )
                    })
                }
            }

            return out
        }

        // ----------------------------
        // Canonical reader layer (V2)
        // ----------------------------

        private object StepSemanticReaderV2 {

            fun readLeadApplication(stepObj: JSONObject): JSONObject? {
                fun containsCellIndex(arr: JSONArray?, cellIndex: Int): Boolean {
                    if (arr == null || cellIndex !in 0..80) return false
                    for (i in 0 until arr.length()) {
                        val obj = arr.optJSONObject(i) ?: continue
                        if (obj.optInt("cellIndex", -1) == cellIndex) return true
                    }
                    return false
                }

                fun containsPlacementCell(arr: JSONArray?, cellIndex: Int): Boolean {
                    if (arr == null || cellIndex !in 0..80) return false
                    for (i in 0 until arr.length()) {
                        val obj = arr.optJSONObject(i) ?: continue
                        val cell = obj.optJSONObject("cell") ?: continue
                        if (cell.optInt("cellIndex", -1) == cellIndex) return true
                    }
                    return false
                }

                val proof = stepObj.optJSONObject("proof")

                proof?.optJSONObject("lead_application")?.let { return it }

                val apps = proof?.optJSONArray("applications")
                if (apps != null && apps.length() > 0) {
                    var first: JSONObject? = null

                    val tech = stepObj.optJSONObject("technique") ?: JSONObject()
                    val stepFamily = tech.optString("family").lowercase()
                    val stepTechniqueId = tech.optString("technique_id")
                        .ifBlank { tech.optString("id") }
                        .ifBlank { tech.optString("app_name") }
                        .lowercase()

                    val targetCellIndex =
                        stepObj.optJSONObject("target")
                            ?.optJSONObject("cell")
                            ?.optInt("cellIndex", -1)
                            ?: -1

                    var bestIntersection: JSONObject? = null
                    var bestIntersectionScore = Int.MIN_VALUE

                    for (i in 0 until apps.length()) {
                        val app = apps.optJSONObject(i) ?: continue
                        if (first == null) first = app

                        val role = app.optJSONObject("narrative")?.optString("role").orEmpty()
                        if (role.equals("trigger", ignoreCase = true)) return app

                        val intersectionsStep =
                            stepFamily == "box_line_interaction" ||
                                    stepTechniqueId == "singles-pointing" ||
                                    stepTechniqueId == "singles-boxed"

                        if (intersectionsStep) {
                            val appTechniqueId = app.optJSONObject("identity")
                                ?.optString("technique_id")
                                .orEmpty()
                                .lowercase()

                            val appArchetype = app.optJSONObject("narrative")
                                ?.optString("archetype")
                                .orEmpty()
                                .uppercase()

                            val pattern = app.optJSONObject("pattern") ?: JSONObject()
                            val patternCells = pattern.optJSONObject("cells") ?: JSONObject()
                            val effects = app.optJSONObject("effects") ?: JSONObject()

                            var score = 0
                            if (appTechniqueId == stepTechniqueId && stepTechniqueId.isNotBlank()) score += 100
                            if (appArchetype == "INTERSECTIONS") score += 50

                            if (targetCellIndex in 0..80) {
                                if (containsCellIndex(patternCells.optJSONArray("focus_cells"), targetCellIndex)) score += 20
                                if (containsCellIndex(patternCells.optJSONArray("pattern_cells"), targetCellIndex)) score += 20
                                if (containsCellIndex(patternCells.optJSONArray("target_cells"), targetCellIndex)) score += 20
                                if (containsPlacementCell(effects.optJSONArray("placements"), targetCellIndex)) score += 15
                                if (containsPlacementCell(effects.optJSONArray("cell_value_forces"), targetCellIndex)) score += 15
                            }

                            if (score > bestIntersectionScore) {
                                bestIntersectionScore = score
                                bestIntersection = app
                            }
                        }
                    }

                    if (bestIntersection != null) return bestIntersection
                    if (first != null) return first
                }

                // Additive fallback for older / audit-facing shapes.
                stepObj.optJSONObject("lead_application")?.let { return it }

                return null
            }

            fun readArchetype(stepObj: JSONObject, leadApp: JSONObject?): NarrativeArchetypeV1 {
                val appArch = leadApp?.optJSONObject("narrative")?.optString("archetype").orEmpty()
                if (appArch.isNotBlank()) {
                    return NarrativeArchetypeV1.fromWire(appArch)
                }

                val tech = stepObj.optJSONObject("technique") ?: JSONObject()
                val family = tech.optString("family").lowercase()
                val techId = tech.optString("technique_id")
                    .ifBlank { tech.optString("id") }
                    .ifBlank { tech.optString("app_name") }
                    .lowercase()
                val techName = tech.optString("technique_name")
                    .ifBlank { tech.optString("name") }
                    .ifBlank { tech.optString("real_name") }
                    .lowercase()

                return when {
                    family == "single" && (techId == "singles-1" || techName.contains("full house") || techName.contains("last remaining cell")) ->
                        NarrativeArchetypeV1.FULL_HOUSE
                    family == "single" && (techId.contains("naked") || techName.contains("naked")) ->
                        NarrativeArchetypeV1.NAKED_SINGLES
                    family == "single" ->
                        NarrativeArchetypeV1.HIDDEN_SINGLES
                    family.contains("multiple") || family.contains("subset") ->
                        NarrativeArchetypeV1.SUBSETS
                    family.contains("box_line") || family.contains("interaction") ->
                        NarrativeArchetypeV1.INTERSECTIONS
                    family.contains("fish") ->
                        NarrativeArchetypeV1.FISH
                    family.contains("wing") || family.contains("boxed_pattern") ->
                        NarrativeArchetypeV1.WINGS
                    family.contains("chain") || family.contains("ring") ->
                        NarrativeArchetypeV1.CHAINS
                    techId.contains("pointing") || techId.contains("boxed") && techId.contains("single") ->
                        NarrativeArchetypeV1.INTERSECTIONS
                    techId.contains("pair") || techId.contains("triple") || techId.contains("quad") ->
                        NarrativeArchetypeV1.SUBSETS
                    techId.contains("x-wing") || techId.contains("xwings") || techId.contains("swordfish") || techId.contains("jellyfish") ->
                        NarrativeArchetypeV1.FISH
                    techId.contains("wing") ->
                        NarrativeArchetypeV1.WINGS
                    techId.contains("chain") || techId.contains("remote") || techId.contains("ring") ->
                        NarrativeArchetypeV1.CHAINS
                    else -> NarrativeArchetypeV1.UNKNOWN
                }
            }

            fun pickPrimaryHouse(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                tr: Int,
                tc: Int,
                tIdx: Int
            ): JSONObject {
                val pattern = leadApp?.optJSONObject("pattern")

                val houseFromPattern = firstHouse(pattern?.optJSONArray("houses"))
                if (houseFromPattern != null) return houseFromPattern

                val houseFromUnitsScanned = firstHouse(pattern?.optJSONArray("units_scanned"))
                if (houseFromUnitsScanned != null) return houseFromUnitsScanned

                val dim = stepObj.optString("dimension")
                    .ifBlank { stepObj.optJSONObject("selected_placement")?.optString("dimension").orEmpty() }
                    .lowercase()

                return when {
                    dim.contains("col") -> houseRef("col", tc)
                    dim.contains("box") -> houseRef("box", boxIndex(tr, tc))
                    dim.contains("row") -> houseRef("row", tr)
                    else -> {
                        val r = (tIdx / 9) + 1
                        houseRef("row", r)
                    }
                }
            }

            fun extractCanonicalHiddenSingle(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                fallbackTargetIdx: Int,
                fallbackDigit: Int
            ): HiddenSingleCanonical? {
                val app = leadApp ?: return null
                val support = app.optJSONObject("support") ?: JSONObject()
                val narrative = app.optJSONObject("narrative") ?: JSONObject()
                val pattern = app.optJSONObject("pattern") ?: JSONObject()
                val effects = app.optJSONObject("effects") ?: JSONObject()

                val archetype = NarrativeArchetypeV1.fromWire(narrative.optString("archetype"))
                if (archetype != NarrativeArchetypeV1.HIDDEN_SINGLES) return null

                val placement = firstPlacement(effects.optJSONArray("placements")) ?: return null
                val focusCell = placement.cell
                val focusIdx = focusCell.cellIndex
                val digit = placement.digit

                val primaryHouse =
                    firstHouse(pattern.optJSONArray("houses"))
                        ?: firstHouse(pattern.optJSONArray("units_scanned"))
                        ?: houseRef("row", focusCell.r)

                val links = support.optJSONArray("explanation_links") ?: JSONArray()
                val pairs = mutableListOf<Pair<CellRef, CellRef>>()

                for (i in 0 until links.length()) {
                    val link = links.optJSONObject(i) ?: continue
                    val kind = link.optString("kind")
                    if (!kind.equals("peer_witness", ignoreCase = true)) continue

                    val peer = parseCellRef(link.opt("peer_cell")) ?: continue
                    val witness = parseCellRef(link.opt("witness_cell")) ?: continue
                    pairs += peer to witness
                }

                if (pairs.isEmpty()) return null

                return HiddenSingleCanonical(
                    primaryHouse = primaryHouse,
                    digit = digit.takeIf { it in 1..9 } ?: fallbackDigit,
                    focusIdx = focusIdx.takeIf { it in 0..80 } ?: fallbackTargetIdx,
                    focusR = focusCell.r,
                    focusC = focusCell.c,
                    peerWitnessPairs = pairs.sortedBy { it.first.cellIndex }
                )
            }

            fun extractCanonicalFullHouse(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                grid81: String?,
                fallbackTargetIdx: Int,
                fallbackDigit: Int
            ): FullHouseCanonical? {
                val app = leadApp ?: return null
                val narrative = app.optJSONObject("narrative") ?: JSONObject()
                val pattern = app.optJSONObject("pattern") ?: JSONObject()
                val effects = app.optJSONObject("effects") ?: JSONObject()
                val grid = (grid81 ?: "").takeIf { it.length == 81 } ?: return null

                val archetype = NarrativeArchetypeV1.fromWire(narrative.optString("archetype"))
                if (archetype != NarrativeArchetypeV1.FULL_HOUSE) return null

                val placement = firstPlacement(effects.optJSONArray("placements")) ?: return null
                val focusCell = placement.cell
                val focusIdx = focusCell.cellIndex
                val digit = placement.digit.takeIf { it in 1..9 } ?: fallbackDigit

                val primaryHouse =
                    firstHouse(pattern.optJSONArray("houses"))
                        ?: firstHouse(pattern.optJSONArray("units_scanned"))
                        ?: houseRef("row", focusCell.r)

                val houseIndices = houseCellIndicesOrEmpty(primaryHouse)
                if (houseIndices.isEmpty()) return null

                val emptyIndices = houseIndices.filter { idx -> grid[idx] == '.' || grid[idx] == '0' }
                if (emptyIndices.size != 1) return null

                val remainingIdx = emptyIndices.first()
                if (remainingIdx != focusIdx && remainingIdx != fallbackTargetIdx) return null

                val usedDigits =
                    houseIndices.mapNotNull { idx ->
                        grid[idx].digitToIntOrNull()?.takeIf { it in 1..9 }
                    }.distinct().sorted()

                val missingDigits = (1..9).filter { it !in usedDigits }
                if (missingDigits.size != 1) return null

                val remainingDigit = missingDigits.first()
                if (remainingDigit != digit) return null

                return FullHouseCanonical(
                    primaryHouse = primaryHouse,
                    digit = digit,
                    focusIdx = focusIdx.takeIf { it in 0..80 } ?: fallbackTargetIdx,
                    focusR = focusCell.r,
                    focusC = focusCell.c,
                    remainingCell = CellRef(focusCell.r, focusCell.c, focusIdx.takeIf { it in 0..80 } ?: fallbackTargetIdx),
                    remainingDigit = remainingDigit,
                    filledDigits = usedDigits
                )
            }

            fun extractCanonicalNakedSingle(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                fallbackTargetIdx: Int,
                fallbackDigit: Int
            ): NakedSingleCanonical? {
                val app = leadApp ?: return null
                val support = app.optJSONObject("support") ?: JSONObject()
                val narrative = app.optJSONObject("narrative") ?: JSONObject()
                val pattern = app.optJSONObject("pattern") ?: JSONObject()
                val effects = app.optJSONObject("effects") ?: JSONObject()

                val archetype = NarrativeArchetypeV1.fromWire(narrative.optString("archetype"))
                if (archetype != NarrativeArchetypeV1.NAKED_SINGLES) return null

                val placement = firstPlacement(effects.optJSONArray("placements")) ?: return null
                val focusCell = placement.cell
                val focusIdx = focusCell.cellIndex
                val digit = placement.digit

                val links = support.optJSONArray("explanation_links") ?: JSONArray()
                val pairs = mutableListOf<Pair<Int, CellRef>>()
                val eliminatedDigits = mutableSetOf<Int>()

                for (i in 0 until links.length()) {
                    val link = links.optJSONObject(i) ?: continue
                    val kind = link.optString("kind")
                    if (!kind.equals("digit_witness", ignoreCase = true)) continue

                    val elimDigit = link.optInt("eliminated_digit", -1)
                    val witness = parseCellRef(link.opt("witness_cell")) ?: continue
                    if (elimDigit in 1..9) {
                        pairs += elimDigit to witness
                        eliminatedDigits += elimDigit
                    }
                }

                val primaryHouse = JSONObject().apply {
                    put("type", "cell")
                    put("cell", focusCell.toJson())
                }

                val dimensionHouses = parseHouseList(support.optJSONArray("dimension_houses"))
                val defaultCandidateDigits = parseIntList(support.optJSONArray("default_candidate_digits"))
                    .filter { it in 1..9 }
                    .distinct()
                    .sorted()

                if (pairs.isEmpty()) return null

                return NakedSingleCanonical(
                    primaryHouse = primaryHouse,
                    focusIdx = focusIdx.takeIf { it in 0..80 } ?: fallbackTargetIdx,
                    focusR = focusCell.r,
                    focusC = focusCell.c,
                    digit = digit.takeIf { it in 1..9 } ?: fallbackDigit,
                    dimensionHouses = dimensionHouses,
                    defaultCandidateDigits = defaultCandidateDigits,
                    eliminatedDigits = eliminatedDigits.toList().sorted(),
                    digitWitnessPairs = pairs.sortedBy { it.first }
                )
            }

            fun extractCanonicalIntersection(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                fallbackTargetCell: CellRef?,
                fallbackTargetDigit: Int?
            ): CanonicalIntersection? {
                val app = leadApp ?: return null
                val support = app.optJSONObject("support") ?: JSONObject()
                val effects = app.optJSONObject("effects") ?: JSONObject()
                val pattern = app.optJSONObject("pattern") ?: JSONObject()

                val coverSets = pattern.optJSONArray("cover_sets") ?: JSONArray()
                val cover0 = coverSets.optJSONObject(0) ?: JSONObject()

                val interactionKind =
                    support.optString("interaction_kind")
                        .ifBlank { cover0.optString("interaction_kind") }
                        .ifBlank { pattern.optString("pattern_subtype") }
                        .ifBlank { "intersection" }

                val digit = support.optInt(
                    "digit",
                    cover0.optInt("digit", fallbackTargetDigit ?: -1)
                ).takeIf { it in 1..9 }

                val sourceHouse =
                    parseHouse(support.opt("source_house"))
                        ?: parseHouse(cover0.opt("source_house"))
                        ?: firstHouse(pattern.optJSONArray("houses"))
                        ?: return null

                val targetHouse =
                    parseHouse(support.opt("target_house"))
                        ?: parseHouse(cover0.opt("target_house"))
                        ?: firstHouse(pattern.optJSONArray("units_scanned"))
                        ?: return null

                val boxHouse =
                    parseHouse(support.opt("box_house"))
                        ?: parseHouse(cover0.opt("box_house"))

                val lineHouse =
                    parseHouse(support.opt("line_house"))
                        ?: parseHouse(cover0.opt("line_house"))

                val lineType =
                    support.optString("line_type")
                        .ifBlank { cover0.optString("line_type") }
                        .ifBlank { null }

                val orientation =
                    support.optString("orientation")
                        .ifBlank { cover0.optString("orientation") }
                        .ifBlank { null }

                val lockedCells =
                    parseCellList(support.optJSONArray("locked_cells"))
                        .ifEmpty { parseCellList(cover0.optJSONArray("locked_cells")) }
                        .ifEmpty { parseCellList(cover0.optJSONArray("constrained_cells")) }

                val supportSweepCells =
                    parseCellList(support.optJSONArray("sweep_cells"))
                        .ifEmpty { parseCellList(cover0.optJSONArray("sweep_cells")) }

                val immediateElims =
                    parseCandidateElims(effects.optJSONArray("candidate_eliminations"))
                        .filter { digit == null || it.digit == digit }

                val elimSweepCells =
                    immediateElims.map { it.cell }
                        .distinctBy { it.cellIndex }
                        .sortedBy { it.cellIndex }

                val sweepCells =
                    supportSweepCells.ifEmpty { elimSweepCells }

                val witnessCells =
                    parseCellList(support.optJSONArray("witness_cells"))
                        .ifEmpty { lockedCells }

                val explanationLinks = buildList {
                    val arr = support.optJSONArray("explanation_links") ?: JSONArray()
                    for (i in 0 until arr.length()) {
                        val obj = arr.optJSONObject(i) ?: continue
                        add(
                            CanonicalIntersectionExplanationLink(
                                kind = obj.optString("kind").ifBlank { "intersection_witness" },
                                digit = obj.optInt("digit", -1).takeIf { it in 1..9 } ?: digit,
                                interactionKind = obj.optString("interaction_kind").ifBlank { interactionKind },
                                lockedCells = parseCellList(obj.optJSONArray("locked_cells")).ifEmpty { lockedCells },
                                sweepCell = parseCellRef(obj.opt("sweep_cell")),
                                sourceHouse = parseHouse(obj.opt("source_house")) ?: parseHouse(support.opt("source_house")),
                                targetHouse = parseHouse(obj.opt("target_house")) ?: parseHouse(support.opt("target_house")),
                                boxHouse = parseHouse(obj.opt("box_house")) ?: parseHouse(support.opt("box_house")),
                                lineHouse = parseHouse(obj.opt("line_house")) ?: parseHouse(support.opt("line_house")),
                                lineType = obj.optString("line_type").ifBlank { lineType },
                                orientation = obj.optString("orientation").ifBlank { orientation }
                            )
                        )
                    }
                }

                val finalProof = support.optJSONObject("final_canonical_proof") ?: JSONObject()
                val finalProofPayload = finalProof.optJSONObject("proof_payload") ?: JSONObject()
                val finalHouseClaim = finalProofPayload.optJSONObject("house_claim") ?: JSONObject()
                val finalCellOutcome = finalProofPayload.optJSONObject("cell_outcome") ?: JSONObject()

                val sourceConfinementProof = support.optJSONObject("source_confinement_proof")

                val downstreamResolutionKind =
                    finalProof.optString("elimination_kind")
                        .ifBlank { "CELL_CANDIDATE_DIGITS" }

                val downstreamFocusCell =
                    parseCellRef(finalProof.opt("focus_cell"))
                        ?: parseCellRef(finalCellOutcome.opt("cell"))
                        ?: fallbackTargetCell

                val downstreamDigit =
                    finalProof.optInt("digit", -1).takeIf { it in 1..9 }
                        ?: finalHouseClaim.optInt("digit", -1).takeIf { it in 1..9 }
                        ?: finalCellOutcome.optInt("digit", -1).takeIf { it in 1..9 }
                        ?: fallbackTargetDigit
                        ?: digit

                val downstreamPrimaryHouse =
                    parseHouse(finalProof.opt("primary_house"))
                        ?: parseHouse(finalHouseClaim.opt("house"))
                        ?: downstreamFocusCell?.let { cellHouseRef(it.cellIndex) }
                        ?: JSONObject(targetHouse.toString())

                val focusCell =
                    downstreamFocusCell
                        ?: sweepCells.firstOrNull()
                        ?: lockedCells.firstOrNull()

                val primaryHouse = JSONObject(targetHouse.toString())

                return CanonicalIntersection(
                    interactionKind = interactionKind,
                    digit = digit,
                    sourceHouse = JSONObject(sourceHouse.toString()),
                    targetHouse = JSONObject(targetHouse.toString()),
                    boxHouse = boxHouse?.let { JSONObject(it.toString()) },
                    lineHouse = lineHouse?.let { JSONObject(it.toString()) },
                    lineType = lineType,
                    orientation = orientation,
                    lockedCells = lockedCells.distinctBy { it.cellIndex }.sortedBy { it.cellIndex },
                    sweepCells = sweepCells.distinctBy { it.cellIndex }.sortedBy { it.cellIndex },
                    witnessCells = witnessCells.distinctBy { it.cellIndex }.sortedBy { it.cellIndex },
                    explanationLinks = explanationLinks,
                    sourceConfinementProof = sourceConfinementProof?.let { JSONObject(it.toString()) },
                    immediateEliminations = immediateElims.sortedBy { it.cell.cellIndex },
                    downstreamResolutionKind = downstreamResolutionKind,
                    focusCell = focusCell,
                    targetDigit = downstreamDigit,
                    primaryHouse = primaryHouse,
                    finalResolutionKind = downstreamResolutionKind,
                    finalPrimaryHouse = JSONObject(downstreamPrimaryHouse.toString()),
                    finalFocusCell = focusCell,
                    finalTargetDigit = downstreamDigit,
                    finalCanonicalProof = if (finalProof.length() > 0) JSONObject(finalProof.toString()) else null
                )
            }



            fun extractFishData(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                td: Int
            ): FishData? {
                val app = leadApp ?: return null
                val pattern = app.optJSONObject("pattern") ?: return null
                val effects = app.optJSONObject("effects") ?: JSONObject()

                val cs = pattern.optJSONArray("cover_sets")?.optJSONObject(0) ?: return null
                val fishKind = cs.optString("fish_kind").ifBlank { pattern.optString("pattern_subtype") }
                val digit = cs.optInt("digit", td)

                val baseHouses = parseHouseList(cs.optJSONArray("base_houses"))
                val coverHouses = parseHouseList(cs.optJSONArray("cover_houses"))
                val fishCells = parseCellIndexList(cs.optJSONArray("fish_cells"))
                val corners = if (fishCells.isNotEmpty()) fishCells else parseCellIndexList(pattern.optJSONObject("cells")?.optJSONArray("pattern_cells"))
                val sweep = parseCandidateElims(effects.optJSONArray("candidate_eliminations"))
                    .filter { it.digit == digit }
                    .map { it.cell.cellIndex }
                    .distinct()

                if (baseHouses.isEmpty() || coverHouses.isEmpty()) return null

                val baseType = baseHouses.first().optString("type")
                val coverType = coverHouses.first().optString("type")
                val baseIndices = baseHouses.mapNotNull { it.optInt("index1to9", -1).takeIf { n -> n in 1..9 } }
                val coverIndices = coverHouses.mapNotNull { it.optInt("index1to9", -1).takeIf { n -> n in 1..9 } }

                return FishData(
                    fishKind = fishKind,
                    digit = digit,
                    baseType = baseType,
                    baseIndices = baseIndices,
                    coverType = coverType,
                    coverIndices = coverIndices,
                    corners = corners,
                    sweepCells = sweep
                )
            }

            fun extractWingData(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                td: Int
            ): WingData? {
                val app = leadApp ?: return null
                val pattern = app.optJSONObject("pattern") ?: return null
                val effects = app.optJSONObject("effects") ?: JSONObject()
                val roles = pattern.optJSONObject("roles") ?: JSONObject()

                val pivot = firstCell(roles.optJSONArray("pivot"))
                    ?: firstCell(roles.optJSONArray("hinge"))
                    ?: firstCell(pattern.optJSONObject("cells")?.optJSONArray("focus_cells"))
                    ?: return null

                val wingCells =
                    parseCellList(roles.optJSONArray("wing"))
                        .ifEmpty { parseCellList(roles.optJSONArray("pincer")) }
                        .ifEmpty { parseCellList(pattern.optJSONObject("cells")?.optJSONArray("pattern_cells")).filter { it.cellIndex != pivot.cellIndex } }

                val elim = parseCandidateElims(effects.optJSONArray("candidate_eliminations"))
                val eliminationDigit = elim.firstOrNull()?.digit ?: td
                val eliminationTargets = elim.map { it.cell.cellIndex }.distinct()

                if (wingCells.isEmpty() || eliminationTargets.isEmpty()) return null

                return WingData(
                    digit = eliminationDigit,
                    hinge = pivot.cellIndex,
                    pincers = wingCells.map { it.cellIndex }.distinct(),
                    targetEliminate = eliminationTargets.first()
                )
            }

            fun extractChainData(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                td: Int
            ): ChainData? {
                val app = leadApp ?: return null
                val pattern = app.optJSONObject("pattern") ?: return null
                val effects = app.optJSONObject("effects") ?: JSONObject()

                val cs = pattern.optJSONArray("cover_sets")?.optJSONObject(0) ?: return null
                val chainNodesArr = cs.optJSONArray("chain_nodes") ?: return null

                val chainNodes = mutableListOf<Int>()
                for (i in 0 until chainNodesArr.length()) {
                    val node = chainNodesArr.optJSONObject(i) ?: continue
                    val cell = parseCellRef(node.opt("cell")) ?: continue
                    chainNodes += cell.cellIndex
                }
                if (chainNodes.size < 2) return null

                val colorA = mutableListOf<Int>()
                val colorB = mutableListOf<Int>()
                for (i in chainNodes.indices) {
                    if (i % 2 == 0) colorA += chainNodes[i] else colorB += chainNodes[i]
                }

                val elim = parseCandidateElims(effects.optJSONArray("candidate_eliminations"))
                val eliminateCell = elim.firstOrNull()?.cell?.cellIndex

                return ChainData(
                    digit = elim.firstOrNull()?.digit ?: td,
                    colorA = colorA,
                    colorB = colorB,
                    contradictionCell = null,
                    eliminateCell = eliminateCell
                )
            }

            fun extractSubsetData(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                primaryHouseFallback: JSONObject
            ): SubsetData? {
                val app = leadApp ?: return null
                val pattern = app.optJSONObject("pattern") ?: return null
                val effects = app.optJSONObject("effects") ?: JSONObject()

                val subtype = pattern.optString("pattern_subtype").lowercase()
                val mode = when {
                    subtype.startsWith("hidden_") -> "hidden"
                    subtype.startsWith("naked_") -> "naked"
                    else -> inferSubsetMode(app)
                }

                val house = firstHouse(pattern.optJSONArray("houses"))
                    ?: firstHouse(pattern.optJSONArray("units_scanned"))
                    ?: primaryHouseFallback

                val roles = pattern.optJSONObject("roles") ?: JSONObject()
                val subsetCells =
                    parseCellIndexList(roles.optJSONArray("subset_member"))
                        .ifEmpty { parseCellIndexList(roles.optJSONArray("supporting_cell")) }
                        .ifEmpty { parseCellIndexList(pattern.optJSONObject("cells")?.optJSONArray("pattern_cells")) }

                val targetCells =
                    parseCellIndexList(roles.optJSONArray("target"))
                        .ifEmpty { parseCellIndexList(pattern.optJSONObject("cells")?.optJSONArray("target_cells")) }
                        .ifEmpty {
                            if (mode == "naked") {
                                parseCandidateElims(effects.optJSONArray("candidate_eliminations")).map { it.cell.cellIndex }
                            } else {
                                parseCandidateRestrictions(effects.optJSONArray("candidate_restrictions")).map { it.cell.cellIndex }
                            }
                        }

                val lockedDigits =
                    parseIntList(pattern.optJSONArray("digits"))
                        .ifEmpty { parseIntList(roles.optJSONArray("digit")) }
                        .ifEmpty { parseIntList(pattern.optJSONObject("summary")?.optJSONArray("digits")) }
                        .distinct()
                        .sorted()

                val restrictedDigits =
                    parseCandidateRestrictions(effects.optJSONArray("candidate_restrictions"))
                        .flatMap { it.removedDigits }
                        .distinct()
                        .sorted()

                if (subsetCells.isEmpty() || lockedDigits.isEmpty()) return null

                return SubsetData(
                    house = house,
                    subsetMode = mode,
                    subsetSubtype = subtype,
                    subsetCells = subsetCells.distinct(),
                    lockedDigits = lockedDigits,
                    sweepCells = targetCells.distinct(),
                    restrictedDigits = restrictedDigits
                )
            }

            fun extractCanonicalSubset(
                stepObj: JSONObject,
                leadApp: JSONObject?,
                fallbackPrimaryHouse: JSONObject,
                fallbackTargetCell: CellRef?,
                fallbackTargetDigit: Int?
            ): CanonicalSubset? {
                val app = leadApp ?: return null
                val subset = extractSubsetData(stepObj, leadApp, fallbackPrimaryHouse) ?: return null

                val support = app.optJSONObject("support") ?: JSONObject()
                val effects = app.optJSONObject("effects") ?: JSONObject()

                val eliminationKind = support.optString("elimination_kind")
                    .ifBlank {
                        if (subset.subsetMode == "naked") "CELL_CANDIDATE_DIGITS" else "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"
                    }

                val primaryHouse =
                    parseHouse(support.opt("primary_house"))
                        ?: if (eliminationKind == "CELL_CANDIDATE_DIGITS" && fallbackTargetCell != null) {
                            JSONObject().apply {
                                put("type", "cell")
                                put("cell", fallbackTargetCell.toJson())
                            }
                        } else {
                            JSONObject(subset.house.toString())
                        }

                val focusCell =
                    parseCellRef(support.opt("focus_cell"))
                        ?: fallbackTargetCell

                val placements = parsePlacements(effects.optJSONArray("placements"))
                val forces = parsePlacements(effects.optJSONArray("cell_value_forces"))
                val targetDigit =
                    fallbackTargetDigit?.takeIf { it in 1..9 }
                        ?: placements.firstOrNull()?.digit
                        ?: forces.firstOrNull()?.digit
                        ?: support.optInt("digit", -1).takeIf { it in 1..9 }

                val defaultCandidateDigits =
                    parseIntList(support.optJSONArray("default_candidate_digits"))
                        .filter { it in 1..9 }
                        .distinct()
                        .sorted()

                val claimedCandidateDigits =
                    parseIntList(support.optJSONArray("claimed_candidate_digits"))
                        .filter { it in 1..9 }
                        .distinct()
                        .sorted()

                val remainingCandidateDigits =
                    parseIntList(support.optJSONArray("remaining_candidate_digits"))
                        .filter { it in 1..9 }
                        .distinct()
                        .sorted()

                val defaultCandidateCells =
                    parseCellList(support.optJSONArray("default_candidate_cells"))
                        .distinctBy { it.cellIndex }
                        .sortedBy { it.cellIndex }

                val claimedCandidateCells =
                    parseCellList(support.optJSONArray("claimed_candidate_cells"))
                        .distinctBy { it.cellIndex }
                        .sortedBy { it.cellIndex }

                val remainingCandidateCells =
                    parseCellList(support.optJSONArray("remaining_candidate_cells"))
                        .distinctBy { it.cellIndex }
                        .sortedBy { it.cellIndex }

                val witnessCells =
                    parseCellList(support.optJSONArray("witness_cells"))
                        .distinctBy { it.cellIndex }
                        .sortedBy { it.cellIndex }

                val explanationLinks = support.optJSONArray("explanation_links") ?: JSONArray()

                val digitWitnesses = mutableListOf<CanonicalSubsetDigitWitness>()
                val cellWitnesses = mutableListOf<CanonicalSubsetCellWitness>()

                for (i in 0 until explanationLinks.length()) {
                    val link = explanationLinks.optJSONObject(i) ?: continue
                    when (link.optString("kind")) {
                        "digit_witness" -> {
                            val digit = link.optInt("eliminated_digit", -1)
                            if (digit !in 1..9) continue

                            val witnessKind = link.optString("witness_kind").ifBlank { "unknown" }
                            val witnessCell = parseCellRef(link.opt("witness_cell"))
                            val subsetKind = link.optString("subset_kind").takeIf { it.isNotBlank() }
                            val subsetDigits = parseIntList(link.optJSONArray("digits")).filter { it in 1..9 }.distinct().sorted()
                            val subsetCells = parseCellList(link.optJSONArray("cells")).distinctBy { it.cellIndex }.sortedBy { it.cellIndex }
                            val subsetHouse = parseHouse(link.opt("house"))

                            digitWitnesses += CanonicalSubsetDigitWitness(
                                digit = digit,
                                witnessKind = witnessKind,
                                witnessCell = witnessCell,
                                relation = link.optString("relation").takeIf { it.isNotBlank() },
                                subsetKind = subsetKind,
                                subsetDigits = subsetDigits,
                                subsetCells = subsetCells,
                                subsetHouse = subsetHouse
                            )
                        }

                        "peer_witness" -> {
                            val claimedCell = parseCellRef(link.opt("peer_cell")) ?: continue
                            val witnessKind = link.optString("witness_kind").ifBlank { "unknown" }
                            val witnessCell = parseCellRef(link.opt("witness_cell"))
                            val subsetKind = link.optString("subset_kind").takeIf { it.isNotBlank() }
                            val subsetDigits = parseIntList(link.optJSONArray("digits")).filter { it in 1..9 }.distinct().sorted()
                            val subsetCells = parseCellList(link.optJSONArray("cells")).distinctBy { it.cellIndex }.sortedBy { it.cellIndex }
                            val subsetHouse = parseHouse(link.opt("house"))

                            cellWitnesses += CanonicalSubsetCellWitness(
                                claimedCell = claimedCell,
                                witnessKind = witnessKind,
                                witnessCell = witnessCell,
                                relation = link.optString("relation").takeIf { it.isNotBlank() },
                                subsetKind = subsetKind,
                                subsetDigits = subsetDigits,
                                subsetCells = subsetCells,
                                subsetHouse = subsetHouse
                            )
                        }
                    }
                }

                return CanonicalSubset(
                    subset = subset,
                    eliminationKind = eliminationKind,
                    primaryHouse = primaryHouse,
                    focusCell = focusCell,
                    targetDigit = targetDigit,
                    defaultCandidateDigits = defaultCandidateDigits,
                    claimedCandidateDigits = claimedCandidateDigits,
                    remainingCandidateDigits = remainingCandidateDigits,
                    defaultCandidateCells = defaultCandidateCells,
                    claimedCandidateCells = claimedCandidateCells,
                    remainingCandidateCells = remainingCandidateCells,
                    witnessCells = witnessCells,
                    digitWitnesses = digitWitnesses.sortedBy { it.digit },
                    cellWitnesses = cellWitnesses.sortedBy { it.claimedCell.cellIndex }
                )
            }

            private fun inferSubsetMode(app: JSONObject): String {
                val techId = app.optJSONObject("identity")?.optString("technique_id").orEmpty().lowercase()
                return if (techId.contains("doubles") || techId.contains("triplets") || techId.contains("quads")) {
                    "hidden"
                } else {
                    "naked"
                }
            }
        }

        // ----------------------------
        // Data classes
        // ----------------------------

        private data class CellRef(val r: Int, val c: Int, val cellIndex: Int) {
            fun toJson(): JSONObject = JSONObject().apply {
                put("r", r)
                put("c", c)
                put("cellIndex", cellIndex)
            }
        }

        private data class Placement(val cell: CellRef, val digit: Int)
        private data class CandidateElim(val cell: CellRef, val digit: Int)
        private data class CandidateRestriction(val cell: CellRef, val removedDigits: List<Int>, val remainingDigits: List<Int>)

        private data class HiddenSingleCanonical(
            val primaryHouse: JSONObject,
            val digit: Int,
            val focusIdx: Int,
            val focusR: Int,
            val focusC: Int,
            val peerWitnessPairs: List<Pair<CellRef, CellRef>>
        )

        private data class NakedSingleCanonical(
            val primaryHouse: JSONObject,
            val focusIdx: Int,
            val focusR: Int,
            val focusC: Int,
            val digit: Int,
            val dimensionHouses: List<JSONObject>,
            val defaultCandidateDigits: List<Int>,
            val eliminatedDigits: List<Int>,
            val digitWitnessPairs: List<Pair<Int, CellRef>>
        )

        private data class FullHouseCanonical(
            val primaryHouse: JSONObject,
            val digit: Int,
            val focusIdx: Int,
            val focusR: Int,
            val focusC: Int,
            val remainingCell: CellRef,
            val remainingDigit: Int,
            val filledDigits: List<Int>
        )

        private data class IntersectionData(
            val digit: Int,
            val focusIdx: Int,
            val focusR: Int,
            val focusC: Int,
            val sourceHouse: JSONObject,
            val targetHouse: JSONObject,
            val interactionKind: String,
            val constrainedCells: List<Int>,
            val sweepCells: List<Int>
        )

        private data class CanonicalIntersectionExplanationLink(
            val kind: String,
            val digit: Int?,
            val interactionKind: String?,
            val lockedCells: List<CellRef>,
            val sweepCell: CellRef?,
            val sourceHouse: JSONObject? = null,
            val targetHouse: JSONObject? = null,
            val boxHouse: JSONObject? = null,
            val lineHouse: JSONObject? = null,
            val lineType: String? = null,
            val orientation: String? = null
        )

        private data class CanonicalIntersection(
            val interactionKind: String,
            val digit: Int?,
            val sourceHouse: JSONObject,
            val targetHouse: JSONObject,
            val boxHouse: JSONObject?,
            val lineHouse: JSONObject?,
            val lineType: String?,
            val orientation: String?,
            val lockedCells: List<CellRef>,
            val sweepCells: List<CellRef>,
            val witnessCells: List<CellRef>,
            val explanationLinks: List<CanonicalIntersectionExplanationLink>,
            val sourceConfinementProof: JSONObject?,
            val immediateEliminations: List<CandidateElim>,
            val downstreamResolutionKind: String,
            val focusCell: CellRef?,
            val targetDigit: Int?,
            val primaryHouse: JSONObject,
            val finalResolutionKind: String,
            val finalPrimaryHouse: JSONObject,
            val finalFocusCell: CellRef?,
            val finalTargetDigit: Int?,
            val finalCanonicalProof: JSONObject?
        )

        private data class SubsetData(
            val house: JSONObject,
            val subsetMode: String,          // naked / hidden
            val subsetSubtype: String,       // naked_pair / hidden_pair / ...
            val subsetCells: List<Int>,
            val lockedDigits: List<Int>,
            val sweepCells: List<Int>,
            val restrictedDigits: List<Int>
        )

        private data class CanonicalSubsetDigitWitness(
            val digit: Int,
            val witnessKind: String,         // single_cell / subset_group / unknown
            val witnessCell: CellRef? = null,
            val relation: String? = null,
            val subsetKind: String? = null,
            val subsetDigits: List<Int> = emptyList(),
            val subsetCells: List<CellRef> = emptyList(),
            val subsetHouse: JSONObject? = null
        )

        private data class CanonicalSubsetCellWitness(
            val claimedCell: CellRef,
            val witnessKind: String,         // single_cell / subset_group / unknown
            val witnessCell: CellRef? = null,
            val relation: String? = null,
            val subsetKind: String? = null,
            val subsetDigits: List<Int> = emptyList(),
            val subsetCells: List<CellRef> = emptyList(),
            val subsetHouse: JSONObject? = null
        )

        private data class CanonicalSubset(
            val subset: SubsetData,
            val eliminationKind: String,
            val primaryHouse: JSONObject,
            val focusCell: CellRef?,
            val targetDigit: Int?,
            val defaultCandidateDigits: List<Int>,
            val claimedCandidateDigits: List<Int>,
            val remainingCandidateDigits: List<Int>,
            val defaultCandidateCells: List<CellRef>,
            val claimedCandidateCells: List<CellRef>,
            val remainingCandidateCells: List<CellRef>,
            val witnessCells: List<CellRef>,
            val digitWitnesses: List<CanonicalSubsetDigitWitness>,
            val cellWitnesses: List<CanonicalSubsetCellWitness>
        )

        private data class FishData(
            val fishKind: String,
            val digit: Int,
            val baseType: String,
            val baseIndices: List<Int>,
            val coverType: String,
            val coverIndices: List<Int>,
            val corners: List<Int>,
            val sweepCells: List<Int>
        )

        private data class WingData(
            val digit: Int,
            val hinge: Int,
            val pincers: List<Int>,
            val targetEliminate: Int
        )

        private data class ChainData(
            val digit: Int,
            val colorA: List<Int>,
            val colorB: List<Int>,
            val contradictionCell: Int? = null,
            val eliminateCell: Int? = null
        )

        private fun isBaseArchetypeForAtom0Invariant(archetype: NarrativeArchetypeV1): Boolean =
            archetype == NarrativeArchetypeV1.HIDDEN_SINGLES ||
                    archetype == NarrativeArchetypeV1.NAKED_SINGLES ||
                    archetype == NarrativeArchetypeV1.FULL_HOUSE

        private fun introCellLabelV1(cell: JSONObject?): String {
            val r = cell?.optInt("r", -1) ?: -1
            val c = cell?.optInt("c", -1) ?: -1
            return if (r > 0 && c > 0) "row $r, column $c" else "the target cell"
        }

        private fun introHouseLabelV1(house: JSONObject?): String {
            val type = house?.optString("type").orEmpty()
            val index = house?.optInt("index1to9", -1) ?: -1
            return when (type) {
                "row" -> if (index > 0) "row $index" else "this row"
                "col" -> if (index > 0) "column $index" else "this column"
                "box" -> if (index > 0) "box $index" else "this box"
                "cell" -> introCellLabelV1(house?.optJSONObject("cell"))
                else -> "this area"
            }
        }

        private fun introDigitsLabelV1(digits: JSONArray?): String {
            if (digits == null || digits.length() == 0) return "these digits"
            val values = mutableListOf<String>()
            for (i in 0 until digits.length()) {
                val value = digits.optInt(i, -1)
                if (value > 0) values += value.toString()
            }
            return when (values.size) {
                0 -> "these digits"
                1 -> values[0]
                2 -> "${values[0]} and ${values[1]}"
                else -> values.dropLast(1).joinToString(", ") + ", and " + values.last()
            }
        }

        private fun introCellsLabelV1(cells: JSONArray?): String {
            if (cells == null || cells.length() == 0) return "these cells"
            val labels = mutableListOf<String>()
            for (i in 0 until cells.length()) {
                val cell = cells.optJSONObject(i)
                labels += introCellLabelV1(cell)
            }
            return when (labels.size) {
                0 -> "these cells"
                1 -> labels[0]
                2 -> "${labels[0]} and ${labels[1]}"
                else -> labels.dropLast(1).joinToString(", ") + ", and " + labels.last()
            }
        }

        private fun introTechniqueNameV1(archetype: NarrativeArchetypeV1, triggerPattern: JSONObject): String =
            when (archetype) {
                NarrativeArchetypeV1.SUBSETS -> {
                    val subtype = triggerPattern.optString("subset_subtype").ifBlank { "subset" }
                    subtype.replace('_', ' ')
                }
                NarrativeArchetypeV1.INTERSECTIONS -> {
                    triggerPattern.optString("interaction_kind").ifBlank { "intersection" }
                        .replace('_', ' ')
                }
                else -> archetype.wire.lowercase().replace('_', ' ')
            }

        private fun buildAdvancedIntroOverlayContractV1(
            archetype: NarrativeArchetypeV1
        ): JSONObject = JSONObject().apply {
            put("schema_version", "advanced_intro_overlay_contract_v1")
            put("intent", "SHOW_SPOTLIGHT")
            put("show_target_focus", true)

            when (archetype) {
                NarrativeArchetypeV1.SUBSETS -> {
                    put("setup_variant", "subset_pattern_tableau")
                    put("show_subset_house", true)
                    put("show_subset_cells", true)
                    put("show_subset_candidates", true)
                    // North Star intro:
                    // show why the pair exists, and why it matters for the target.
                    put("show_sweep_cells", true)
                    put("show_blocker_network", true)
                }
                NarrativeArchetypeV1.INTERSECTIONS -> {
                    put("setup_variant", "intersection_crossroads_tableau")

                    // Core territorial geometry
                    put("show_source_house", true)
                    put("show_cross_house", true)
                    put("show_box_house", true)
                    put("show_line_house", true)

                    // The overlap itself must be visually explicit.
                    put("show_overlap_cells", true)
                    put("show_pattern_cells", true)

                    // New native intersection setup cue:
                    // show the trapped digit inside the surviving overlap cells.
                    put("show_confined_overlap_digits", true)

                    // Setup law: audit the source house outside the overlap.
                    put("show_source_outside_overlap_cells", true)
                    put("show_source_outside_audit_witnesses", true)

                    // Preview the territorial consequence, but do not spend the target yet.
                    put("show_forbidden_cross_cells_preview", true)

                    // Backward-compatible flags
                    put("show_target_house", true)
                    put("show_sweep_cells", false)
                    put("show_blocker_network", false)
                }
                else -> {
                    put("setup_variant", "advanced_pattern_tableau")
                    put("show_sweep_cells", false)
                    put("show_blocker_network", false)
                }
            }

            put("show_resolution_collapse", false)
            put("show_commit", false)
        }

        private fun applyIntroOverlayContractV1(
            dst: JSONObject,
            frameId: String,
            setupRole: String,
            contract: JSONObject?
        ): JSONObject {
            dst.put(
                "intent",
                contract?.optString("intent")
                    ?.takeIf { it.isNotBlank() }
                    ?: OverlayIntentV1.SHOW_SPOTLIGHT.wire
            )
            dst.put("frame_id", frameId)
            dst.put("setup_role", setupRole)

            contract?.keys()?.forEach { key ->
                if (key == "schema_version" || key == "intent") return@forEach
                dst.put(key, contract.opt(key))
            }

            return dst
        }

        private fun buildAdvancedIntroDerivedFieldsV1(
            archetype: NarrativeArchetypeV1,
            resolutionKind: String,
            targetCell: JSONObject,
            primaryHouse: JSONObject,
            triggerPattern: JSONObject,
            triggerExplanation: JSONObject,
            triggerBridge: JSONObject
        ): JSONObject = JSONObject().apply {
            fun groupedWitnessExplanationForMemberProof(proof: JSONObject): String {
                val witnessByDigit =
                    proof.optJSONArray("witness_by_digit") ?: JSONArray()

                val rowDigits = mutableListOf<Int>()
                val colDigits = mutableListOf<Int>()
                val boxDigits = mutableListOf<Int>()
                val otherDigits = mutableListOf<Int>()

                for (i in 0 until witnessByDigit.length()) {
                    val row = witnessByDigit.optJSONObject(i) ?: continue
                    val digit = row.optInt("digit", -1)
                    if (digit !in 1..9) continue

                    val viaHouse = row.optJSONObject("via_house")
                    when (viaHouse?.optString("type").orEmpty()) {
                        "row" -> rowDigits += digit
                        "col" -> colDigits += digit
                        "box" -> boxDigits += digit
                        else -> otherDigits += digit
                    }
                }

                fun digitsLabel(ds: List<Int>): String =
                    when (ds.distinct().sorted().size) {
                        0 -> ""
                        1 -> ds.distinct().sorted()[0].toString()
                        2 -> ds.distinct().sorted().joinToString(" and ")
                        else -> {
                            val vals = ds.distinct().sorted()
                            vals.dropLast(1).joinToString(", ") + ", and " + vals.last()
                        }
                    }

                val clauses = mutableListOf<String>()

                if (rowDigits.isNotEmpty()) {
                    val ds = rowDigits.distinct().sorted()
                    clauses += if (ds.size == 1) {
                        "${digitsLabel(ds)} is blocked from the row"
                    } else {
                        "${digitsLabel(ds)} are blocked from the row"
                    }
                }

                if (colDigits.isNotEmpty()) {
                    val ds = colDigits.distinct().sorted()
                    clauses += if (ds.size == 1) {
                        "${digitsLabel(ds)} is blocked from the column"
                    } else {
                        "${digitsLabel(ds)} are blocked from the column"
                    }
                }

                if (boxDigits.isNotEmpty()) {
                    val ds = boxDigits.distinct().sorted()
                    clauses += if (ds.size == 1) {
                        "${digitsLabel(ds)} is already accounted for in the box"
                    } else {
                        "${digitsLabel(ds)} are already accounted for in the box"
                    }
                }

                if (otherDigits.isNotEmpty()) {
                    val ds = otherDigits.distinct().sorted()
                    clauses += if (ds.size == 1) {
                        "${digitsLabel(ds)} is blocked by another visible witness"
                    } else {
                        "${digitsLabel(ds)} are blocked by other visible witnesses"
                    }
                }

                return when (clauses.size) {
                    0 -> "Every other candidate is already ruled out by visible witnesses around the cell."
                    1 -> clauses[0] + "."
                    2 -> clauses[0] + ", and " + clauses[1] + "."
                    else -> clauses.dropLast(1).joinToString(", ") + ", and " + clauses.last() + "."
                }
            }

            val targetLabel = introCellLabelV1(targetCell)
            val targetHouseLabel = introHouseLabelV1(primaryHouse)
            val techniqueName = introTechniqueNameV1(archetype, triggerPattern)

            put(
                "narrative_route",
                JSONArray().apply {
                    put("trigger_explanation")
                    put("trigger")
                    put("bridge")
                    put("final_resolution_setup")
                }
            )

            put(
                "target_orientation_summary",
                "Let's zoom in on $targetLabel."
            )

            when (archetype) {
                NarrativeArchetypeV1.SUBSETS -> {
                    val subsetHouse = triggerPattern.optJSONObject("house") ?: primaryHouse
                    val subsetHouseLabel = introHouseLabelV1(subsetHouse)
                    val subsetCells = triggerPattern.optJSONArray("subset_cells")
                    val lockedDigits = triggerPattern.optJSONArray("locked_digits")
                    val subsetCellsLabel = introCellsLabelV1(subsetCells)
                    val lockedDigitsLabel = introDigitsLabelV1(lockedDigits)

                    val memberProofs =
                        triggerExplanation.optJSONArray("pattern_member_proofs")
                            ?: triggerExplanation.optJSONArray("member_proofs")
                            ?: JSONArray()

                    put(
                        "technique_lens_summary",
                        "This cell is a good moment for a $techniqueName idea, not because the answer is obvious right away, but because $subsetHouseLabel is starting to organize itself in a very useful way."
                    )

                    put(
                        "trigger_explanation_summary",
                        "Inside $subsetHouseLabel, two cells are each squeezed down to the same two digits: $lockedDigitsLabel."
                    )

                    val memberRows = JSONArray()
                    for (i in 0 until memberProofs.length()) {
                        val proof = memberProofs.optJSONObject(i) ?: continue
                        val cellLabel = introCellLabelV1(proof.optJSONObject("cell"))
                        val remainingDigits =
                            introDigitsLabelV1(proof.optJSONArray("remaining_candidate_digits"))
                        val explanation = groupedWitnessExplanationForMemberProof(proof)

                        val introLead = when (i) {
                            0 -> "Here is the first half of that pattern: $cellLabel."
                            1 -> "Now look at $cellLabel."
                            else -> "Look at $cellLabel."
                        }

                        val collapseLine = when (i) {
                            0 -> "That cell gets narrowed all the way down to $remainingDigits."
                            1 -> "The same thing happens again. This cell also collapses to $remainingDigits."
                            else -> "This cell also collapses to $remainingDigits."
                        }

                        val reasonLead = if (i == 0) {
                            "The reason is that "
                        } else {
                            "Everything else is ruled out by its own witnesses: "
                        }

                        memberRows.put(
                            JSONObject().apply {
                                put("member_cell", proof.opt("cell") ?: JSONObject.NULL)
                                put(
                                    "surviving_digits",
                                    proof.optJSONArray("remaining_candidate_digits") ?: JSONArray()
                                )
                                put("grouped_witness_summary", explanation)
                                put(
                                    "spoken_line",
                                    listOf(
                                        introLead,
                                        collapseLine,
                                        reasonLead + explanation
                                    ).joinToString(" ").trim()
                                )
                            }
                        )
                    }
                    put("trigger_member_explanation_rows", memberRows)

                    put(
                        "trigger_summary",
                        "So now we have something very specific: $subsetCellsLabel hold exactly $lockedDigitsLabel inside $subsetHouseLabel. Two cells in the same house, holding exactly the same two candidates. That is a $techniqueName."
                    )

                    put(
                        "bridge_summary",
                        "And once that pair appears, those two digits are effectively reserved for those two cells, which means the rest of $subsetHouseLabel has to let go of $lockedDigitsLabel. That is why this matters for our target $targetLabel: this pair is exactly the kind of setup that can clean it up and make it solvable."
                    )

                    put(
                        "final_resolution_setup_summary",
                        "So this is a setup move for $targetLabel, not the final placement yet."
                    )

                    put(
                        "honesty_note",
                        "The $techniqueName does not place the answer by itself; it sets up the target cell."
                    )
                }

                NarrativeArchetypeV1.INTERSECTIONS -> {
                    val sourceHouse = triggerPattern.optJSONObject("source_house")
                    val targetHouse = triggerPattern.optJSONObject("target_house")
                    val sourceLabel = introHouseLabelV1(sourceHouse)
                    val targetLabelHouse = introHouseLabelV1(targetHouse)

                    put(
                        "technique_lens_summary",
                        "This looks like a good moment for an intersection idea."
                    )

                    put(
                        "trigger_explanation_summary",
                        "A digit is confined in one house strongly enough to constrain its overlap with another house."
                    )

                    put(
                        "trigger_summary",
                        "This looks like an intersection pattern between $sourceLabel and $targetLabelHouse."
                    )

                    put(
                        "bridge_summary",
                        triggerBridge.optString("why_this_matters").ifBlank {
                            "That interaction can clean up the target without finishing the step yet."
                        }
                    )

                    put(
                        "final_resolution_setup_summary",
                        "So this is about setting up the target, not committing the answer yet."
                    )

                    put(
                        "honesty_note",
                        "The intersection pattern prepares the target cell; it does not finish the move by itself."
                    )
                }

                else -> {
                    put(
                        "technique_lens_summary",
                        "This looks like a good moment for an advanced pattern."
                    )
                    put(
                        "trigger_explanation_summary",
                        "There is an advanced pattern here that makes this area of the grid worth inspecting."
                    )
                    put(
                        "trigger_summary",
                        "That pattern is the key to this step."
                    )
                    put(
                        "bridge_summary",
                        triggerBridge.optString("why_this_matters").ifBlank {
                            "That pattern matters because it helps clean up the eventual target."
                        }
                    )
                    put(
                        "final_resolution_setup_summary",
                        if (resolutionKind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT") {
                            "So this setup narrows the house before the final resolution."
                        } else {
                            "So this setup narrows the target cell before the final resolution."
                        }
                    )
                    put(
                        "honesty_note",
                        "This pattern sets up the answer rather than placing it immediately."
                    )
                }
            }

            put("cta_kind", "SHOW_PROOF")
            put("target_house_label", targetHouseLabel)
        }

        private fun buildAtom0InvariantContractV1(
            archetype: NarrativeArchetypeV1,
            resolutionKind: String,
            targetCell: JSONObject,
            primaryHouse: JSONObject,
            triggerPattern: JSONObject? = null,
            triggerExplanation: JSONObject? = null,
            triggerBridge: JSONObject? = null
        ): JSONObject = JSONObject().apply {
            put("schema_version", "atom0_invariant_v1")

            put("target_alignment", JSONObject().apply {
                put("focus_cell", JSONObject(targetCell.toString()))
                put("primary_house", JSONObject(primaryHouse.toString()))
                put("resolution_kind", resolutionKind)
                put("requires_focus_cell_match", true)
                put("requires_primary_house_match_when_house_based", true)
            })

            val isAdvanced = !isBaseArchetypeForAtom0Invariant(archetype)
            put("advanced_trigger_required", isAdvanced)

            put("intro_alignment", JSONObject().apply {
                put("is_intro_anchor", true)
                put("requires_single_walkthrough_cta", true)
                put(
                    "required_narrative_route",
                    JSONArray().apply {
                        put("trigger_explanation")
                        put("trigger")
                        put("bridge")
                        put("final_resolution_setup")
                    }
                )
                put("must_include_bounded_trigger_member_explanation", true)
                put("must_not_resolve_final_answer_in_setup", true)
            })

            if (isAdvanced) {
                put("advanced_trigger", JSONObject().apply {
                    put("pattern_required", true)
                    put("explanation_required", true)
                    put("bridge_required", true)
                    put("intro_summary_fields_required", true)

                    put(
                        "pattern",
                        triggerPattern?.let { JSONObject(it.toString()) }
                            ?: JSONObject().apply { put("status", "pending_population") }
                    )

                    put(
                        "explanation",
                        triggerExplanation?.let { JSONObject(it.toString()) }
                            ?: JSONObject().apply { put("status", "pending_population") }
                    )

                    put(
                        "bridge_to_target",
                        triggerBridge?.let { JSONObject(it.toString()) }
                            ?: JSONObject().apply { put("status", "pending_population") }
                    )
                })

                put(
                    "overlay_alignment",
                    buildAdvancedIntroOverlayContractV1(archetype)
                )
            }
        }

        private fun buildAdvancedAtom0SetupPayloadV1(
            archetype: NarrativeArchetypeV1,
            resolutionKind: String,
            targetCell: JSONObject,
            primaryHouse: JSONObject,
            triggerPattern: JSONObject,
            triggerExplanation: JSONObject,
            triggerBridge: JSONObject
        ): JSONObject = JSONObject().apply {
            val introDerived =
                buildAdvancedIntroDerivedFieldsV1(
                    archetype = archetype,
                    resolutionKind = resolutionKind,
                    targetCell = targetCell,
                    primaryHouse = primaryHouse,
                    triggerPattern = triggerPattern,
                    triggerExplanation = triggerExplanation,
                    triggerBridge = triggerBridge
                )

            val introOverlayContract = buildAdvancedIntroOverlayContractV1(archetype)

            put("setup_role", "advanced_trigger_setup")
            put("archetype", archetype.wire)
            put("resolution_kind", resolutionKind)
            put("target_cell", JSONObject(targetCell.toString()))
            put("primary_house", JSONObject(primaryHouse.toString()))
            put("trigger_pattern", JSONObject(triggerPattern.toString()))
            put("trigger_explanation", JSONObject(triggerExplanation.toString()))
            put("trigger_bridge", JSONObject(triggerBridge.toString()))

            put("intro_route", JSONArray(introDerived.optJSONArray("narrative_route")?.toString() ?: "[]"))
            put("intro_derived", JSONObject(introDerived.toString()))
            put("intro_overlay_contract", JSONObject(introOverlayContract.toString()))

            put(
                "intro_narration_contract",
                JSONObject().apply {
                    put("schema_version", "advanced_intro_narration_contract_v1")
                    put("must_start_from_target_orientation", true)
                    put("must_name_technique_early", true)
                    put("must_include_trigger_explanation_summary", true)
                    put("must_include_bounded_trigger_member_explanation", true)
                    put("must_include_trigger_summary", true)
                    put("must_include_bridge_summary", true)
                    put("must_include_final_resolution_setup_summary", true)
                    put("must_include_honesty_note", true)
                    put("must_end_with_single_walkthrough_cta", true)
                    put("must_not_resolve_final_answer_in_setup", true)
                    put(
                        "ordered_beats",
                        JSONArray().apply {
                            put("target_orientation")
                            put("technique_lens")
                            put("trigger_explanation")
                            put("trigger")
                            put("bridge")
                            put("final_resolution_setup")
                            put("single_walkthrough_cta")
                        }
                    )
                }
            )

            put(
                "atom0_invariant_contract",
                buildAtom0InvariantContractV1(
                    archetype = archetype,
                    resolutionKind = resolutionKind,
                    targetCell = targetCell,
                    primaryHouse = primaryHouse,
                    triggerPattern = triggerPattern,
                    triggerExplanation = triggerExplanation,
                    triggerBridge = triggerBridge
                )
            )
        }

        // ----------------------------
        // Atom builders
        // ----------------------------


        private fun atomSpotlight(
            index: Int,
            archetype: NarrativeArchetypeV1,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject = JSONObject().apply {
            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", archetype.wire)
            put("beat_kind", NarrativeBeatKindV1.SPOTLIGHT.wire)
            put("spoiler_level", SpoilerLevelV1.NONE.wire)

            val targetCell = cellRef(tr, tc, tIdx)
            val isNaked = archetype == NarrativeArchetypeV1.NAKED_SINGLES
            val contractResolutionKind =
                when (archetype) {
                    NarrativeArchetypeV1.NAKED_SINGLES -> "CELL_CANDIDATE_DIGITS"
                    NarrativeArchetypeV1.HIDDEN_SINGLES,
                    NarrativeArchetypeV1.FULL_HOUSE -> "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"
                    else -> "PENDING_FINAL_RESOLUTION_BINDING"
                }

            put("focus", JSONObject().apply {
                put("target_cell", targetCell)
                put("target_digit", td)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put(
                    "code",
                    if (isNaked) NarrativeClaimCodeV1.SEARCH_DIGITS_IN_CELL.wire
                    else NarrativeClaimCodeV1.SEARCH_DIGIT_IN_HOUSE.wire
                )
                put("args", JSONObject().apply {
                    if (isNaked) {
                        put("cell", JSONObject(targetCell.toString()))
                    } else {
                        put("digit", td)
                        put("house", JSONObject(primaryHouse.toString()))
                    }

                    put(
                        "atom0_invariant_contract",
                        buildAtom0InvariantContractV1(
                            archetype = archetype,
                            resolutionKind = contractResolutionKind,
                            targetCell = targetCell,
                            primaryHouse = primaryHouse,
                            triggerPattern =
                                if (!isBaseArchetypeForAtom0Invariant(archetype)) {
                                    JSONObject().apply {
                                        put("status", "pending_truth_population")
                                        put("archetype", archetype.wire)
                                    }
                                } else null,
                            triggerExplanation =
                                if (!isBaseArchetypeForAtom0Invariant(archetype)) {
                                    JSONObject().apply {
                                        put("status", "pending_truth_population")
                                        put("source", "advanced_atom0_contract_phase1")
                                    }
                                } else null,
                            triggerBridge =
                                if (!isBaseArchetypeForAtom0Invariant(archetype)) {
                                    JSONObject().apply {
                                        put("status", "pending_truth_population")
                                        put("source", "advanced_atom0_contract_phase1")
                                    }
                                } else null
                        )
                    )
                })
            })

            put("witnesses", JSONArray())
            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })
            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_SPOTLIGHT.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }






        private fun buildSubsetAtomsPacketFromTruthV2(
            stepObj: JSONObject,
            grid81: String?,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            fallbackPrimaryHouse: JSONObject
        ): JSONObject {
            val truth = buildNarrativeTruthV2(stepObj, grid81)
            val finalResolution = truth.optJSONObject("final_resolution") ?: JSONObject()
            val witnessPattern = truth.optJSONObject("witness_pattern") ?: JSONObject()
            val proofPayload = truth.optJSONObject("proof_payload") ?: JSONObject()
            val triggerPattern = truth.optJSONObject("trigger_pattern") ?: JSONObject()
            val triggerExplanation = truth.optJSONObject("trigger_explanation") ?: JSONObject()
            val triggerBridge = truth.optJSONObject("trigger_bridge") ?: JSONObject()
            val triggerPacket = truth.optJSONObject("trigger_packet") ?: JSONObject()

            val truthPrimaryHouse = truth.optJSONObject("primary_house") ?: fallbackPrimaryHouse
            val finalPrimaryHouse =
                parseHouse(finalResolution.opt("primary_house"))
                    ?: truthPrimaryHouse

            val focusObj = truth.optJSONObject("focus") ?: JSONObject()
            val fallbackFocusCell =
                parseCellRef(focusObj.opt("focus_cell"))
                    ?: CellRef(tr, tc, tIdx)

            val fallbackDigit =
                focusObj.optInt("digit", td).takeIf { it in 1..9 } ?: td

            val resolutionKind =
                finalResolution.optString("kind")
                    .ifBlank { truth.optString("resolution_kind") }

            val focusCell =
                parseCellRef(finalResolution.opt("focus_cell"))
                    ?: fallbackFocusCell

            val focusDigit =
                finalResolution.optInt("digit", fallbackDigit)
                    .takeIf { it in 1..9 }
                    ?: fallbackDigit

            val atoms = JSONArray()
            var nextIndex = 0

            atoms.put(
                atomSubsetSpotlightFromTruthV2(
                    index = nextIndex++,
                    focusCell = focusCell,
                    focusDigit = focusDigit,
                    primaryHouse = finalPrimaryHouse,
                    resolutionKind = resolutionKind,
                    witnessPattern = witnessPattern,
                    proofPayload = proofPayload,
                    triggerPattern = triggerPattern,
                    triggerExplanation = triggerExplanation,
                    triggerBridge = triggerBridge,
                    triggerPacket = triggerPacket
                )
            )

            if (witnessPattern.length() > 0) {
                atoms.put(
                    atomSubsetPatternFromTruthV2(
                        index = nextIndex++,
                        focusCell = focusCell,
                        focusDigit = focusDigit,
                        primaryHouse = finalPrimaryHouse,
                        resolutionKind = resolutionKind,
                        witnessPattern = witnessPattern,
                        proofPayload = proofPayload
                    )
                )

                val support = proofPayload.optJSONObject("support") ?: JSONObject()
                val witnessByDigit = support.optJSONArray("witness_by_digit") ?: JSONArray()
                if (resolutionKind == "CELL_CANDIDATE_DIGITS" && witnessByDigit.length() > 0) {
                    atoms.put(
                        atomSubsetReceiptsFromTruthV2(
                            index = nextIndex++,
                            focusCell = focusCell,
                            focusDigit = focusDigit,
                            primaryHouse = finalPrimaryHouse,
                            resolutionKind = resolutionKind,
                            witnessPattern = witnessPattern,
                            proofPayload = proofPayload
                        )
                    )
                }
            } else {
                atoms.put(
                    atomTeachingNoteSubset(
                        nextIndex++,
                        focusCell.r,
                        focusCell.c,
                        focusCell.cellIndex,
                        focusDigit,
                        finalPrimaryHouse
                    )
                )
            }

            val houseClaim = proofPayload.optJSONObject("house_claim") ?: JSONObject()
            val cellOutcome = proofPayload.optJSONObject("cell_outcome") ?: JSONObject()

            val lockCell =
                parseCellRef(finalResolution.opt("focus_cell"))
                    ?: run {
                        val remainingCells = parseCellIndexList(houseClaim.optJSONArray("remaining_candidate_cells"))
                        if (remainingCells.isNotEmpty()) {
                            val idx = remainingCells.first()
                            CellRef((idx / 9) + 1, (idx % 9) + 1, idx)
                        } else {
                            focusCell
                        }
                    }

            val lockDigit =
                finalResolution.optInt("digit", -1).takeIf { it in 1..9 }
                    ?: parseIntList(cellOutcome.optJSONArray("remaining_candidate_digits")).firstOrNull { it in 1..9 }
                    ?: houseClaim.optInt("digit", -1).takeIf { it in 1..9 }
                    ?: focusDigit

            val eliminatedPeers =
                parseCellIndexList(houseClaim.optJSONArray("claimed_candidate_cells"))

            val eliminatedDigits =
                parseIntList(cellOutcome.optJSONArray("claimed_candidate_digits"))

            atoms.put(
                atomLockInFromTruthV2(
                    index = nextIndex++,
                    focusCell = lockCell,
                    focusDigit = lockDigit,
                    primaryHouse = finalPrimaryHouse,
                    resolutionKind = resolutionKind,
                    eliminatedPeers = if (resolutionKind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT") eliminatedPeers else emptyList(),
                    eliminatedDigits = if (resolutionKind == "CELL_CANDIDATE_DIGITS") eliminatedDigits else emptyList(),
                    proofPayload = proofPayload
                )
            )

            atoms.put(
                atomCommit(
                    index = nextIndex,
                    archetype = NarrativeArchetypeV1.SUBSETS,
                    tr = lockCell.r,
                    tc = lockCell.c,
                    tIdx = lockCell.cellIndex,
                    td = lockDigit,
                    primaryHouse = finalPrimaryHouse
                )
            )

            val packet = JSONObject().apply {
                put("schema_version", "narrative_packet_v1")
                put("evidence", JSONObject().apply {
                    put("narrative_truth_v2", JSONObject(truth.toString()))
                    put("narrative_atoms_v1", JSONObject().apply {
                        put("schema_version", "narrative_atoms_v1")
                        put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
                        put("atoms", JSONArray(atoms.toString()))
                        put("final_resolution", JSONObject(finalResolution.toString()))
                        put("version_note", "canonical_reader_v2_truth_v2_subsets_final_resolution_aligned_stage_scope_validated_trigger_packet_audited")
                    })
                })
            }

            val problems = JSONArray()
            ProofValidatorV1.validateFinalResolutionContract(
                packet = packet,
                atoms = atoms,
                finalResolution = finalResolution,
                problems = problems
            )

            val audit = JSONObject().apply {
                put("kind", "NARRATIVE_PACKET_AUDIT_V2")
                put("atom0_snapshot", ProofValidatorV1.buildAtom0AuditSnapshotV2(packet))
                put("summary", JSONObject().apply {
                    put("validation_problem_count", problems.length())
                    put("validation_status", if (problems.length() > 0) "invalid" else "ok")
                })
            }

            return JSONObject().apply {
                put("schema_version", "narrative_atoms_v1")
                put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
                put("atoms", atoms)
                put("final_resolution", JSONObject(finalResolution.toString()))
                put("audit", audit)
                put(
                    "validation",
                    JSONObject().apply {
                        put("status", if (problems.length() > 0) "invalid" else "ok")
                        put("problems", problems)
                    }
                )
                put("version_note", "canonical_reader_v2_truth_v2_subsets_final_resolution_aligned_stage_scope_validated_trigger_packet_audited")
            }
        }



        private fun buildIntersectionAtomsPacketFromTruthV2(
            stepObj: JSONObject,
            grid81: String?,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            fallbackPrimaryHouse: JSONObject
        ): JSONObject {
            val truth = buildNarrativeTruthV2(stepObj, grid81)
            val finalResolution = truth.optJSONObject("final_resolution") ?: JSONObject()
            val witnessPattern = truth.optJSONObject("witness_pattern") ?: JSONObject()
            val proofPayload = truth.optJSONObject("proof_payload") ?: JSONObject()
            val triggerPattern = truth.optJSONObject("trigger_pattern") ?: JSONObject()
            val triggerExplanation = truth.optJSONObject("trigger_explanation") ?: JSONObject()
            val triggerBridge = truth.optJSONObject("trigger_bridge") ?: JSONObject()
            val triggerPacket = truth.optJSONObject("trigger_packet") ?: JSONObject()

            val truthPrimaryHouse = truth.optJSONObject("primary_house") ?: fallbackPrimaryHouse
            val finalPrimaryHouse =
                parseHouse(finalResolution.opt("primary_house"))
                    ?: truthPrimaryHouse

            val focusObj = truth.optJSONObject("focus") ?: JSONObject()
            val fallbackFocusCell =
                parseCellRef(focusObj.opt("focus_cell"))
                    ?: CellRef(tr, tc, tIdx)

            val fallbackDigit =
                focusObj.optInt("digit", td).takeIf { it in 1..9 } ?: td

            val finalResolutionKind =
                finalResolution.optString("kind")
                    .ifBlank { "CELL_CANDIDATE_DIGITS" }

            val focusCell =
                parseCellRef(finalResolution.opt("focus_cell"))
                    ?: fallbackFocusCell

            val focusDigit =
                finalResolution.optInt("digit", fallbackDigit)
                    .takeIf { it in 1..9 }
                    ?: fallbackDigit

            val atoms = JSONArray()
            var nextIndex = 0

            atoms.put(
                atomIntersectionSpotlightFromTruthV2(
                    index = nextIndex++,
                    focusCell = focusCell,
                    focusDigit = focusDigit,
                    primaryHouse = finalPrimaryHouse,
                    resolutionKind = finalResolutionKind,
                    witnessPattern = witnessPattern,
                    triggerPattern = triggerPattern,
                    triggerExplanation = triggerExplanation,
                    triggerBridge = triggerBridge,
                    triggerPacket = triggerPacket
                )
            )

            val support = proofPayload.optJSONObject("support") ?: JSONObject()
            val sourceConfinementProof =
                triggerExplanation.optJSONObject("source_confinement_proof")
                    ?.takeIf { it.length() > 0 }
                    ?: support.optJSONObject("source_confinement_proof")
                    ?: JSONObject()

            if (sourceConfinementProof.length() > 0) {
                atoms.put(
                    atomIntersectionSourceConfinementFromTruthV2(
                        index = nextIndex++,
                        focusCell = focusCell,
                        focusDigit = focusDigit,
                        primaryHouse = finalPrimaryHouse,
                        triggerPattern = triggerPattern,
                        triggerExplanation = triggerExplanation,
                        sourceConfinementProof = sourceConfinementProof
                    )
                )
            }

            if (triggerPattern.length() > 0 || triggerBridge.length() > 0) {
                atoms.put(
                    atomIntersectionPatternFromTruthV2(
                        index = nextIndex++,
                        focusCell = focusCell,
                        focusDigit = focusDigit,
                        primaryHouse = finalPrimaryHouse,
                        triggerPattern = triggerPattern,
                        triggerBridge = triggerBridge
                    )
                )
            }

            val finalCanonicalProof =
                proofPayload.optJSONObject("final_canonical_proof")
                    ?: support.optJSONObject("final_canonical_proof")
                    ?: JSONObject()

            if (finalCanonicalProof.length() > 0) {
                atoms.put(
                    atomIntersectionResolutionFromTruthV2(
                        index = nextIndex++,
                        focusCell = focusCell,
                        focusDigit = focusDigit,
                        primaryHouse = finalPrimaryHouse,
                        finalResolutionKind = finalResolutionKind,
                        finalCanonicalProof = finalCanonicalProof
                    )
                )
            } else {
                atoms.put(
                    atomTeachingNoteIntersection(
                        index = nextIndex++,
                        tr = focusCell.r,
                        tc = focusCell.c,
                        tIdx = focusCell.cellIndex,
                        td = focusDigit,
                        primaryHouse = finalPrimaryHouse
                    )
                )
            }

            atoms.put(
                atomLockIn(
                    index = nextIndex++,
                    archetype = NarrativeArchetypeV1.INTERSECTIONS,
                    tr = focusCell.r,
                    tc = focusCell.c,
                    tIdx = focusCell.cellIndex,
                    td = focusDigit,
                    primaryHouse = finalPrimaryHouse,
                    eliminatedPeers = emptyList(),
                    eliminatedDigits = emptyList()
                )
            )

            atoms.put(
                atomCommit(
                    index = nextIndex,
                    archetype = NarrativeArchetypeV1.INTERSECTIONS,
                    tr = focusCell.r,
                    tc = focusCell.c,
                    tIdx = focusCell.cellIndex,
                    td = focusDigit,
                    primaryHouse = finalPrimaryHouse
                )
            )

            val packet = JSONObject().apply {
                put("schema_version", "narrative_packet_v1")
                put("evidence", JSONObject().apply {
                    put("narrative_truth_v2", JSONObject(truth.toString()))
                    put("narrative_atoms_v1", JSONObject().apply {
                        put("schema_version", "narrative_atoms_v1")
                        put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
                        put("atoms", JSONArray(atoms.toString()))
                        put("final_resolution", JSONObject(finalResolution.toString()))
                        put("version_note", "canonical_reader_v2_truth_v2_intersections_final_resolution_aligned_stage_scope_validated_trigger_packet_audited")
                    })
                })
            }

            val problems = JSONArray()
            ProofValidatorV1.validateFinalResolutionContract(
                packet = packet,
                atoms = atoms,
                finalResolution = finalResolution,
                problems = problems
            )

            val audit = JSONObject().apply {
                put("kind", "NARRATIVE_PACKET_AUDIT_V2")
                put("atom0_snapshot", ProofValidatorV1.buildAtom0AuditSnapshotV2(packet))
                put("summary", JSONObject().apply {
                    put("validation_problem_count", problems.length())
                    put("validation_status", if (problems.length() > 0) "invalid" else "ok")
                })
            }

            return JSONObject().apply {
                put("schema_version", "narrative_atoms_v1")
                put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
                put("atoms", atoms)
                put("final_resolution", JSONObject(finalResolution.toString()))
                put("audit", audit)
                put(
                    "validation",
                    JSONObject().apply {
                        put("status", if (problems.length() > 0) "invalid" else "ok")
                        put("problems", problems)
                    }
                )
                put("version_note", "canonical_reader_v2_truth_v2_intersections_final_resolution_aligned_stage_scope_validated_trigger_packet_audited")
            }
        }

        private fun atomIntersectionSpotlightFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            resolutionKind: String,
            witnessPattern: JSONObject,
            triggerPattern: JSONObject,
            triggerExplanation: JSONObject,
            triggerBridge: JSONObject,
            triggerPacket: JSONObject
        ): JSONObject = JSONObject().apply {
            val interactionKind =
                triggerPattern.optString("interaction_kind")
                    .ifBlank { witnessPattern.optString("interaction_kind") }

            val sourceHouse =
                triggerPattern.optJSONObject("source_house")
                    ?: witnessPattern.optJSONObject("source_house")
                    ?: primaryHouse

            val crossHouse =
                triggerPattern.optJSONObject("cross_house")
                    ?: triggerPattern.optJSONObject("target_house")
                    ?: witnessPattern.optJSONObject("cross_house")
                    ?: witnessPattern.optJSONObject("target_house")
                    ?: primaryHouse

            val overlapCells =
                triggerPattern.optJSONArray("overlap_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: triggerPattern.optJSONArray("locked_cells")
                    ?: witnessPattern.optJSONArray("overlap_cells")
                    ?: witnessPattern.optJSONArray("locked_cells")
                    ?: JSONArray()

            val normalizedTriggerPattern =
                if (triggerPattern.length() > 0) JSONObject(triggerPattern.toString()) else JSONObject().apply {
                    put("kind", "INTERSECTION")
                    put("interaction_kind", interactionKind)
                    put("source_house", JSONObject(sourceHouse.toString()))
                    put("cross_house", JSONObject(crossHouse.toString()))
                    put("target_house", JSONObject(crossHouse.toString())) // backward-compatible alias
                    put("overlap_cells", JSONArray(overlapCells.toString()))
                    put("locked_cells", JSONArray(overlapCells.toString()))
                }

            val normalizedTriggerExplanation =
                if (triggerExplanation.length() > 0) {
                    JSONObject(triggerExplanation.toString())
                } else {
                    JSONObject().apply {
                        put("kind", "INTERSECTION_TRIGGER_EXPLANATION")
                        put("status", "missing_trigger_explanation")
                        put("expected_shape", "source_confinement_proof")
                    }
                }.apply {
                    if (optJSONObject("source_confinement_proof") == null) {
                        put(
                            "source_confinement_proof",
                            JSONObject().apply { put("status", "missing_source_confinement_proof") }
                        )
                    }
                }

            val normalizedTriggerBridge =
                if (triggerBridge.length() > 0) {
                    JSONObject(triggerBridge.toString())
                } else {
                    JSONObject().apply {
                        put("kind", "INTERSECTION_TRIGGER_BRIDGE")
                        put("interaction_kind", interactionKind)
                        put("cross_house_now_restricted", true)
                        put("downstream_resolution_kind", resolutionKind)
                    }
                }.apply {
                    if (optString("downstream_resolution_kind").isBlank() &&
                        optString("final_resolution_kind").isBlank()
                    ) {
                        put("downstream_resolution_kind", resolutionKind)
                    }
                }

            val advancedSetupPayload =
                buildAdvancedAtom0SetupPayloadV1(
                    archetype = NarrativeArchetypeV1.INTERSECTIONS,
                    resolutionKind = resolutionKind,
                    targetCell = focusCell.toJson(),
                    primaryHouse = primaryHouse,
                    triggerPattern = normalizedTriggerPattern,
                    triggerExplanation = normalizedTriggerExplanation,
                    triggerBridge = normalizedTriggerBridge
                ).apply {
                    put(
                        "intersection_setup_identity",
                        JSONObject().apply {
                            put("setup_family", "TERRITORIAL_CONTROL")
                            put("opening_stage_kind", "crossroads_overlap")
                            put("must_explicitly_audit_source_outside_overlap", true)
                            put("must_stop_before_target_effect", true)
                        }
                    )
                }

            val confrontationSummary =
                triggerPacket.optJSONObject("confrontation_summary")
                    ?.let { JSONObject(it.toString()) }
                    ?: JSONObject().apply {
                        put("kind", "INTERSECTION_CONFRONTATION_SUMMARY")
                        put("status", "missing_confrontation_summary")
                        put(
                            "source_confinement_proof",
                            normalizedTriggerExplanation.optJSONObject("source_confinement_proof")
                                ?: JSONObject().apply { put("status", "missing_source_confinement_proof") }
                        )
                    }

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
            put("beat_kind", NarrativeBeatKindV1.SPOTLIGHT.wire)
            put("spoiler_level", SpoilerLevelV1.NONE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put(
                    "code",
                    if (resolutionKind == "CELL_CANDIDATE_DIGITS") {
                        NarrativeClaimCodeV1.SEARCH_DIGITS_IN_CELL.wire
                    } else {
                        NarrativeClaimCodeV1.SEARCH_DIGIT_IN_HOUSE.wire
                    }
                )
                put("args", JSONObject().apply {
                    if (resolutionKind == "CELL_CANDIDATE_DIGITS") {
                        put("cell", focusCell.toJson())
                    } else {
                        put("digit", focusDigit)
                        put("house", primaryHouse)
                    }

                    put("setup_role", "advanced_trigger_setup")
                    put("trigger_pattern", JSONObject(normalizedTriggerPattern.toString()))
                    put("trigger_explanation", JSONObject(normalizedTriggerExplanation.toString()))
                    put("trigger_bridge", JSONObject(normalizedTriggerBridge.toString()))
                    put("trigger_packet", JSONObject(triggerPacket.toString()))
                    put("confrontation_summary", JSONObject(confrontationSummary.toString()))
                    put("advanced_setup_payload", JSONObject(advancedSetupPayload.toString()))
                    put(
                        "atom0_invariant_contract",
                        JSONObject(
                            advancedSetupPayload.optJSONObject("atom0_invariant_contract")?.toString() ?: "{}"
                        )
                    )
                })

                put("meta", JSONObject().apply {
                    put("interaction_kind", interactionKind)
                    put("setup_identity", "INTERSECTION_CROSSROADS_SPOTLIGHT")
                })
            })

            put("witnesses", JSONArray())
            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            val introOverlayContract =
                advancedSetupPayload.optJSONObject("intro_overlay_contract")
                    ?: buildAdvancedIntroOverlayContractV1(NarrativeArchetypeV1.INTERSECTIONS)

            put(
                "overlay",
                applyIntroOverlayContractV1(
                    dst = JSONObject(),
                    frameId = "ov:atom:$index",
                    setupRole = "advanced_trigger_setup",
                    contract = introOverlayContract
                )
            )

            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }



        private fun atomIntersectionSourceConfinementFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            triggerPattern: JSONObject,
            triggerExplanation: JSONObject,
            sourceConfinementProof: JSONObject
        ): JSONObject = JSONObject().apply {
            val sourceHouse =
                sourceConfinementProof.optJSONObject("source_house")
                    ?: triggerPattern.optJSONObject("source_house")
                    ?: primaryHouse

            val crossHouse =
                sourceConfinementProof.optJSONObject("cross_house")
                    ?: triggerPattern.optJSONObject("cross_house")
                    ?: triggerPattern.optJSONObject("target_house")
                    ?: primaryHouse

            val overlapCells =
                sourceConfinementProof.optJSONArray("overlap_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: triggerPattern.optJSONArray("overlap_cells")
                    ?: triggerPattern.optJSONArray("locked_cells")
                    ?: JSONArray()

            val survivingCells =
                parseCellList(
                    triggerExplanation.optJSONArray("overlap_survivor_cells")
                        ?: sourceConfinementProof.optJSONArray("surviving_cells")
                        ?: overlapCells
                )
                    .distinctBy { it.cellIndex }
                    .sortedBy { it.cellIndex }


            val outsideAuditRows =
                triggerExplanation.optJSONArray("setup_preferred_audit_rows")
                    ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_outside_open_seat_audit")
                        ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_outside_overlap_audit")
                        ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("setup_preferred_audit_rows")
                        ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("outside_open_seat_audit_rows")
                        ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("outside_audit_rows")
                    ?: sourceConfinementProof.optJSONArray("eliminated_source_cells")
                    ?: JSONArray()



            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
            put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", sourceHouse)
            })

            put("claim", JSONObject().apply {
                put("code", "INTERSECTION_SOURCE_OUTSIDE_AUDIT")
                put("args", JSONObject().apply {
                    put("digit", sourceConfinementProof.optInt("digit", focusDigit))
                    put("source_house", sourceHouse)
                    put("cross_house", crossHouse)
                    put("semantic_completeness", sourceConfinementProof.optString("semantic_completeness"))
                    put("overlap_cells", JSONArray(overlapCells.toString()))
                    put("surviving_cells", JSONArray().apply {
                        survivingCells.forEach { put(it.toJson()) }
                    })
                    put("outside_audit_rows", JSONArray(outsideAuditRows.toString()))
                    put(
                        "forced_into_overlap_summary",
                        triggerExplanation.opt("forced_into_overlap_summary") ?: JSONObject.NULL
                    )
                })
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", NarrativeWitnessKindV1.OTHER.wire)
                    put("because", JSONObject().apply {
                        put("source_house", sourceHouse)
                        put("cross_house", crossHouse)
                        put("overlap_cells", JSONArray(overlapCells.toString()))
                        put("surviving_cells", JSONArray().apply {
                            survivingCells.forEach { put(it.toJson()) }
                        })
                        put("outside_audit_rows", JSONArray(outsideAuditRows.toString()))
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray().apply {
                    for (i in 0 until outsideAuditRows.length()) {
                        val row = outsideAuditRows.optJSONObject(i) ?: continue
                        val cell = row.optJSONObject("cell") ?: continue
                        val digit = row.optInt("digit", -1)
                        if (digit !in 1..9) continue

                        put(JSONObject().apply {
                            put("cellIndex", cell.optInt("cellIndex", -1))
                            put("digit", digit)
                            put(
                                "reason_code",
                                row.optString("reason_kind")
                                    .ifBlank { "intersection_source_outside_audit" }
                            )
                        })
                    }
                })
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }



        private fun atomIntersectionPatternFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            triggerPattern: JSONObject,
            triggerBridge: JSONObject
        ): JSONObject = JSONObject().apply {
            val interactionKind = triggerPattern.optString("interaction_kind").ifBlank { "intersection" }
            val sourceHouse = triggerPattern.optJSONObject("source_house") ?: primaryHouse
            val crossHouse =
                triggerPattern.optJSONObject("cross_house")
                    ?: triggerPattern.optJSONObject("target_house")
                    ?: primaryHouse
            val overlapCells = parseCellIndexList(
                triggerPattern.optJSONArray("overlap_cells")
                    ?: triggerPattern.optJSONArray("locked_cells")
            )
            val forbiddenCrossCells =
                triggerPattern.optJSONArray("forbidden_cross_cells")
                    ?: triggerBridge.optJSONArray("forbidden_elsewhere_cells")
                    ?: JSONArray()

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
            put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put("code", "INTERSECTION_TERRITORIAL_CONTROL")
                put("args", JSONObject().apply {
                    put("digit", triggerPattern.optInt("digit", focusDigit).takeIf { it in 1..9 } ?: focusDigit)
                    put("source_house", sourceHouse)
                    put("cross_house", crossHouse)
                    put("interaction_kind", interactionKind)
                    put("pattern_subtype", triggerPattern.opt("pattern_subtype") ?: JSONObject.NULL)
                    put("overlap_cells", JSONArray().apply { overlapCells.forEach { put(it) } })
                    put("forbidden_cross_cells", JSONArray(forbiddenCrossCells.toString()))
                    put(
                        "cross_house_permission_change",
                        triggerBridge.opt("cross_house_permission_change") ?: JSONObject.NULL
                    )
                    put(
                        "why_this_matters",
                        triggerBridge.opt("why_this_matters") ?: JSONObject.NULL
                    )
                })
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", NarrativeWitnessKindV1.INTERSECTION_LOCK.wire)
                    put("because", JSONObject().apply {
                        put("overlap_cells", JSONArray().apply { overlapCells.forEach { put(it) } })
                        put("source_house", sourceHouse)
                        put("cross_house", crossHouse)
                        put("interaction_kind", interactionKind)
                        put("box_house", triggerPattern.optJSONObject("box_house") ?: JSONObject.NULL)
                        put("line_house", triggerPattern.optJSONObject("line_house") ?: JSONObject.NULL)
                        put("line_type", triggerPattern.opt("line_type") ?: JSONObject.NULL)
                        put("orientation", triggerPattern.opt("orientation") ?: JSONObject.NULL)
                        put("forbidden_cross_cells", JSONArray(forbiddenCrossCells.toString()))
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }



        private fun atomIntersectionResolutionFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            finalResolutionKind: String,
            finalCanonicalProof: JSONObject
        ): JSONObject = JSONObject().apply {
            val proofPayload = finalCanonicalProof.optJSONObject("proof_payload") ?: JSONObject()
            val houseClaim = proofPayload.optJSONObject("house_claim") ?: JSONObject()
            val cellOutcome = proofPayload.optJSONObject("cell_outcome") ?: JSONObject()

            val remainingCells = parseCellList(houseClaim.optJSONArray("remaining_candidate_cells"))
            val claimedCells = parseCellList(houseClaim.optJSONArray("claimed_candidate_cells"))
            val remainingDigits = parseIntList(cellOutcome.optJSONArray("remaining_candidate_digits"))
            val claimedDigits = parseIntList(cellOutcome.optJSONArray("claimed_candidate_digits"))

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
            put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                if (finalResolutionKind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT") {
                    put("code", "INTERSECTION_DOWNSTREAM_HOUSE_PROOF")
                    put("args", JSONObject().apply {
                        put("digit", houseClaim.optInt("digit", focusDigit).takeIf { it in 1..9 } ?: focusDigit)
                        put("house", parseHouse(houseClaim.opt("house")) ?: primaryHouse)
                        put("remaining_candidate_cells", JSONArray().apply {
                            remainingCells.forEach { put(it.toJson()) }
                        })
                        put("claimed_candidate_cells", JSONArray().apply {
                            claimedCells.forEach { put(it.toJson()) }
                        })
                        put(
                            "default_candidate_cells",
                            JSONArray(houseClaim.optJSONArray("default_candidate_cells")?.toString() ?: "[]")
                        )
                    })
                } else {
                    put("code", "INTERSECTION_DOWNSTREAM_CELL_PROOF")
                    put("args", JSONObject().apply {
                        put("cell", focusCell.toJson())
                        put("remaining_candidate_digits", JSONArray().apply { remainingDigits.forEach { put(it) } })
                        put("claimed_candidate_digits", JSONArray().apply { claimedDigits.forEach { put(it) } })
                    })
                }
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", NarrativeWitnessKindV1.INTERSECTION_LOCK.wire)
                    put("because", JSONObject().apply {
                        put("final_resolution_kind", finalResolutionKind)
                        put(
                            "remaining_candidate_cells",
                            JSONArray().apply { remainingCells.forEach { put(it.toJson()) } }
                        )
                        put(
                            "claimed_candidate_cells",
                            JSONArray().apply { claimedCells.forEach { put(it.toJson()) } }
                        )
                        put(
                            "remaining_candidate_digits",
                            JSONArray().apply { remainingDigits.forEach { put(it) } }
                        )
                        put(
                            "claimed_candidate_digits",
                            JSONArray().apply { claimedDigits.forEach { put(it) } }
                        )
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray().apply {
                    if (finalResolutionKind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT") {
                        claimedCells.forEach { cell ->
                            put(JSONObject().apply {
                                put("cellIndex", cell.cellIndex)
                                put("r", cell.r)
                                put("c", cell.c)
                                put("digit", houseClaim.optInt("digit", focusDigit).takeIf { it in 1..9 } ?: focusDigit)
                                put("reason_code", "intersection_downstream_house_claim")
                            })
                        }
                    } else {
                        claimedDigits.forEach { digit ->
                            put(JSONObject().apply {
                                put("cellIndex", focusCell.cellIndex)
                                put("r", focusCell.r)
                                put("c", focusCell.c)
                                put("digit", digit)
                                put("reason_code", "intersection_downstream_cell_claim")
                            })
                        }
                    }
                })
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_SWEEP.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }









        private fun atomWitnessElimination(
            index: Int,
            archetype: NarrativeArchetypeV1,
            focusIdx: Int,
            focusR: Int,
            focusC: Int,
            digit: Int,
            primaryHouse: JSONObject,
            peerIdx: Int,
            witnessIdx: Int,
            witnessDigit: Int,
            relation: String
        ): JSONObject = JSONObject().apply {
            val pr = (peerIdx / 9) + 1
            val pc = (peerIdx % 9) + 1
            val wr = (witnessIdx / 9) + 1
            val wc = (witnessIdx % 9) + 1

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", archetype.wire)
            put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", cellRef(focusR, focusC, focusIdx))
                put("target_digit", digit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put("code", NarrativeClaimCodeV1.CELL_CANNOT_BE_DIGIT.wire)
                put("args", JSONObject().apply {
                    put("cell", cellRef(pr, pc, peerIdx))
                    put("digit", digit)
                })
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", NarrativeWitnessKindV1.BLOCKS_CANDIDATE.wire)
                    put("because", JSONObject().apply {
                        put("witness_cell", JSONObject().apply {
                            put("r", wr)
                            put("c", wc)
                            put("cellIndex", witnessIdx)
                            put("digit", witnessDigit)
                        })
                        put("relation", relation)
                        put("explains_peer", cellRef(pr, pc, peerIdx))
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray().put(
                    JSONObject().apply {
                        put("cellIndex", peerIdx)
                        put("r", pr)
                        put("c", pc)
                        put("digit", digit)
                        put("reason_code", "witness_blocks_candidate")
                    }
                ))
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }

        private fun atomLockIn(
            index: Int,
            archetype: NarrativeArchetypeV1,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject,
            eliminatedPeers: List<Int>,
            eliminatedDigits: List<Int>
        ): JSONObject = JSONObject().apply {
            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", archetype.wire)
            put("beat_kind", NarrativeBeatKindV1.LOCK_IN.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", cellRef(tr, tc, tIdx))
                put("target_digit", td)
                put("primary_house", primaryHouse)
            })

            val isNaked = archetype == NarrativeArchetypeV1.NAKED_SINGLES
            val isFullHouse = archetype == NarrativeArchetypeV1.FULL_HOUSE
            put("claim", JSONObject().apply {
                put(
                    "code",
                    if (isNaked) NarrativeClaimCodeV1.ONLY_DIGIT_LEFT_FOR_CELL.wire
                    else NarrativeClaimCodeV1.ONLY_CELL_LEFT_FOR_DIGIT_IN_HOUSE.wire
                )
                put("args", JSONObject().apply {
                    if (isNaked) {
                        put("cell", cellRef(tr, tc, tIdx))
                        put("remaining_digits", JSONArray().put(td))
                        put("eliminated_digits", JSONArray().apply { eliminatedDigits.forEach { put(it) } })
                    } else {
                        put("digit", td)
                        put("house", primaryHouse)
                        put("remaining_cells", JSONArray().put(cellRef(tr, tc, tIdx)))
                        if (isFullHouse) {
                            put("remaining_digit", td)
                        }
                    }
                })
            })

            val witnessKind =
                if (isNaked) NarrativeWitnessKindV1.CELL_REMAINS_ONE_DIGIT.wire
                else NarrativeWitnessKindV1.HOUSE_REMAINS_ONE.wire

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", witnessKind)
                    put("because", JSONObject().apply {
                        put("remaining_cell", cellRef(tr, tc, tIdx))
                        if (isNaked) {
                            put("remaining_digit", td)
                            put("eliminated_digits", JSONArray().apply { eliminatedDigits.forEach { put(it) } })
                        } else {
                            if (isFullHouse) {
                                put("remaining_digit", td)
                            }
                            put("eliminated_peer_cell_indices", JSONArray().apply { eliminatedPeers.forEach { put(it) } })
                        }
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_SWEEP.wire)
                put("frame_id", "ov:atom:$index")
            })

            put("user_prompt", JSONObject().apply { put("code", "ASK_USER_TRY") })
        }

        private fun atomCommit(
            index: Int,
            archetype: NarrativeArchetypeV1,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject = JSONObject().apply {
            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", archetype.wire)
            put("beat_kind", NarrativeBeatKindV1.COMMIT.wire)
            put("spoiler_level", SpoilerLevelV1.DIGIT.wire)

            put("focus", JSONObject().apply {
                put("target_cell", cellRef(tr, tc, tIdx))
                put("target_digit", td)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put("code", NarrativeClaimCodeV1.PLACE_DIGIT.wire)
                put("args", JSONObject().apply {
                    put("cell", cellRef(tr, tc, tIdx))
                    put("digit", td)
                })
            })

            put("witnesses", JSONArray())
            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray().put(
                    JSONObject().apply {
                        put("cellIndex", tIdx)
                        put("r", tr)
                        put("c", tc)
                        put("digit", td)
                        put("reason_code", "lock_in")
                    }
                ))
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_COMMIT.wire)
                put("frame_id", "ov:atom:$index")
            })

            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_STEP") })
        }

        // ----------------------------
        // Intersections
        // ----------------------------



        private fun atomTeachingNoteIntersection(
            index: Int,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.INTERSECTIONS.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.NONE.wire)

                put("focus", JSONObject().apply {
                    put("target_cell", cellRef(tr, tc, tIdx))
                    put("target_digit", td)
                    put("primary_house", primaryHouse)
                })

                put("claim", JSONObject().apply {
                    put("code", "TEACHING_NOTE_INTERSECTION")
                    put("args", JSONObject().apply {
                        put(
                            "note",
                            "An intersection pattern traps a digit at the overlap between two houses, then changes what the crossing house is allowed to keep elsewhere."
                        )
                        put("teaching_identity", "TERRITORIAL_CONTROL")
                    })
                })

                put("witnesses", JSONArray())
                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SPOTLIGHT.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        // ----------------------------
        // Subsets
        // ----------------------------

        private fun atomSubsetSpotlightFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            resolutionKind: String,
            witnessPattern: JSONObject,
            proofPayload: JSONObject,
            triggerPattern: JSONObject,
            triggerExplanation: JSONObject,
            triggerBridge: JSONObject,
            triggerPacket: JSONObject
        ): JSONObject = JSONObject().apply {
            val subsetMode = witnessPattern.optString("subset_mode")
            val subsetSubtype = witnessPattern.optString("subset_subtype")
            val subsetHouse = witnessPattern.optJSONObject("house") ?: primaryHouse
            val subsetCells = witnessPattern.optJSONArray("subset_cells") ?: JSONArray()
            val lockedDigits = witnessPattern.optJSONArray("locked_digits") ?: JSONArray()
            val sweepCells = witnessPattern.optJSONArray("sweep_cells") ?: JSONArray()
            val subsetCellCandidates = witnessPattern.optJSONArray("subset_cell_candidates") ?: JSONArray()
            val targetRelation = witnessPattern.optJSONObject("target_relation") ?: JSONObject()
            val sweepRelation = witnessPattern.optJSONObject("sweep_relation") ?: JSONObject()

            val houseClaim = proofPayload.optJSONObject("house_claim") ?: JSONObject()
            val cellOutcome = proofPayload.optJSONObject("cell_outcome") ?: JSONObject()

            val whyThisMatters = targetRelation.optString("why_this_matters").ifBlank {
                when (resolutionKind) {
                    "CELL_CANDIDATE_DIGITS" ->
                        "This subset can clean up the target cell by removing locked digits from contention."
                    "HOUSE_CANDIDATE_CELLS_FOR_DIGIT" ->
                        "This subset can narrow where the digit is allowed to go in the affected house."
                    else ->
                        "This subset changes what remains possible around the target."
                }
            }

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
            put("beat_kind", NarrativeBeatKindV1.SPOTLIGHT.wire)
            put("spoiler_level", SpoilerLevelV1.NONE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            // Phase 2 intent:
            // SETUP must be pattern-centric, not target-centric.
            val normalizedTriggerPattern =
                if (triggerPattern.length() > 0) JSONObject(triggerPattern.toString()) else JSONObject().apply {
                    put("kind", "SUBSET")
                    put("subset_mode", subsetMode)
                    put("subset_subtype", subsetSubtype)
                    put("house", JSONObject(subsetHouse.toString()))
                    put("subset_cells", JSONArray(subsetCells.toString()))
                    put("locked_digits", JSONArray(lockedDigits.toString()))
                    put("subset_cell_candidates", JSONArray(subsetCellCandidates.toString()))
                    put("sweep_cells", JSONArray(sweepCells.toString()))
                }

            val normalizedTriggerExplanation =
                if (triggerExplanation.length() > 0) JSONObject(triggerExplanation.toString()) else JSONObject().apply {
                    put("kind", "SUBSET_TRIGGER_EXPLANATION")
                    put("status", "missing_trigger_explanation")
                    put("expected_shape", "pattern_member_proofs")
                    put("semantic_style", "pattern_member_cell_candidate_proofs")
                    put("intended_kind", "CELL_CANDIDATE_DIGITS")
                }

            val normalizedTriggerBridge =
                if (triggerBridge.length() > 0) JSONObject(triggerBridge.toString()) else JSONObject().apply {
                    put("kind", "SUBSET_TRIGGER_BRIDGE")
                    put("why_this_matters", whyThisMatters)
                    put("target_relation", JSONObject(targetRelation.toString()))
                    put("sweep_relation", JSONObject(sweepRelation.toString()))
                    put("downstream_resolution_kind", resolutionKind)
                }

            val advancedSetupPayload =
                buildAdvancedAtom0SetupPayloadV1(
                    archetype = NarrativeArchetypeV1.SUBSETS,
                    resolutionKind = resolutionKind,
                    targetCell = focusCell.toJson(),
                    primaryHouse = primaryHouse,
                    triggerPattern = normalizedTriggerPattern,
                    triggerExplanation = normalizedTriggerExplanation,
                    triggerBridge = normalizedTriggerBridge
                )

            val confrontationSummary =
                triggerPacket.optJSONObject("confrontation_summary")
                    ?.let { JSONObject(it.toString()) }
                    ?: JSONObject().apply {
                        put("kind", "SUBSET_CONFRONTATION_SUMMARY")
                        put("status", "missing_confrontation_summary")
                    }

            put("claim", JSONObject().apply {
                put("code", NarrativeClaimCodeV1.SUBSET_LOCKS_DIGITS.wire)
                put("args", JSONObject().apply {
                    put("setup_role", "advanced_trigger_setup")
                    put("house", subsetHouse)
                    put("subset_mode", subsetMode)
                    put("subset_subtype", subsetSubtype)
                    put("subset_cells", JSONArray(subsetCells.toString()))
                    put("locked_digits", JSONArray(lockedDigits.toString()))
                    put("subset_cell_candidates", JSONArray(subsetCellCandidates.toString()))
                    put("sweep_cells", JSONArray(sweepCells.toString()))
                    put("target_cell", focusCell.toJson())
                    put("downstream_resolution_kind", resolutionKind)
                    put("target_relation", JSONObject(targetRelation.toString()))
                    put("sweep_relation", JSONObject(sweepRelation.toString()))
                    put("why_this_matters", whyThisMatters)

                    put(
                        "claimed_candidate_digits",
                        JSONArray(cellOutcome.optJSONArray("claimed_candidate_digits")?.toString() ?: "[]")
                    )
                    put(
                        "remaining_candidate_digits",
                        JSONArray(cellOutcome.optJSONArray("remaining_candidate_digits")?.toString() ?: "[]")
                    )
                    put(
                        "claimed_candidate_cells",
                        JSONArray(houseClaim.optJSONArray("claimed_candidate_cells")?.toString() ?: "[]")
                    )
                    put(
                        "remaining_candidate_cells",
                        JSONArray(houseClaim.optJSONArray("remaining_candidate_cells")?.toString() ?: "[]")
                    )

                    put("trigger_pattern", JSONObject(normalizedTriggerPattern.toString()))
                    put("trigger_explanation", JSONObject(normalizedTriggerExplanation.toString()))
                    put("trigger_bridge", JSONObject(normalizedTriggerBridge.toString()))
                    put("trigger_packet", JSONObject(triggerPacket.toString()))
                    put("confrontation_summary", JSONObject(confrontationSummary.toString()))

                    put("advanced_setup_payload", JSONObject(advancedSetupPayload.toString()))
                    put(
                        "atom0_invariant_contract",
                        JSONObject(
                            advancedSetupPayload.optJSONObject("atom0_invariant_contract")?.toString() ?: "{}"
                        )
                    )
                })
            })

            // Phase 6.5:
            // Keep Atom 0 rich, but setup-richness must live in setup-friendly fields
            // (claim.args / focus / overlay), not in witnesses.
            // Confrontation atoms own the actual proof witnesses.
            put("witnesses", JSONArray())

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            val introOverlayContract =
                advancedSetupPayload.optJSONObject("intro_overlay_contract")
                    ?: buildAdvancedIntroOverlayContractV1(NarrativeArchetypeV1.SUBSETS)

            put(
                "overlay",
                applyIntroOverlayContractV1(
                    dst = JSONObject(),
                    frameId = "ov:atom:$index",
                    setupRole = "advanced_trigger_setup",
                    contract = introOverlayContract
                )
            )

            // North Star intro purity:
            // Atom 0 setup semantics must align with the stage-native walkthrough rail.
            // We still let the stage contract own the final visible CTA, but the atom
            // itself must no longer suggest a "try it yourself" branch.
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }

        private fun atomSubsetPatternFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            resolutionKind: String,
            witnessPattern: JSONObject,
            proofPayload: JSONObject
        ): JSONObject = JSONObject().apply {
            val subsetMode = witnessPattern.optString("subset_mode")
            val subsetSubtype = witnessPattern.optString("subset_subtype")
            val subsetHouse = witnessPattern.optJSONObject("house") ?: primaryHouse
            val subsetCells = parseCellIndexList(witnessPattern.optJSONArray("subset_cells"))
            val lockedDigits = parseIntList(witnessPattern.optJSONArray("locked_digits"))
            val subsetCellCandidates = witnessPattern.optJSONArray("subset_cell_candidates") ?: JSONArray()

            val houseClaim = proofPayload.optJSONObject("house_claim") ?: JSONObject()
            val cellOutcome = proofPayload.optJSONObject("cell_outcome") ?: JSONObject()

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
            put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put("code", NarrativeClaimCodeV1.SUBSET_LOCKS_DIGITS.wire)
                put("args", JSONObject().apply {
                    put("house", subsetHouse)
                    put("subset_mode", subsetMode)
                    put("subset_subtype", subsetSubtype)
                    put("subset_cells", JSONArray().apply { subsetCells.forEach { put(it) } })
                    put("locked_digits", JSONArray().apply { lockedDigits.forEach { put(it) } })
                    put("subset_cell_candidates", subsetCellCandidates)

                    put("downstream_resolution_kind", resolutionKind)

                    put(
                        "claimed_candidate_cells",
                        JSONArray(houseClaim.optJSONArray("claimed_candidate_cells")?.toString() ?: "[]")
                    )
                    put(
                        "remaining_candidate_cells",
                        JSONArray(houseClaim.optJSONArray("remaining_candidate_cells")?.toString() ?: "[]")
                    )
                    put(
                        "claimed_candidate_digits",
                        JSONArray(cellOutcome.optJSONArray("claimed_candidate_digits")?.toString() ?: "[]")
                    )
                    put(
                        "remaining_candidate_digits",
                        JSONArray(cellOutcome.optJSONArray("remaining_candidate_digits")?.toString() ?: "[]")
                    )
                })
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", NarrativeWitnessKindV1.SUBSET_DEFINITION.wire)
                    put("because", JSONObject().apply {
                        put("subset_mode", subsetMode)
                        put("subset_subtype", subsetSubtype)
                        put("house", subsetHouse)
                        put("subset_cells", JSONArray().apply { subsetCells.forEach { put(it) } })
                        put("locked_digits", JSONArray().apply { lockedDigits.forEach { put(it) } })
                        put("subset_cell_candidates", subsetCellCandidates)
                        put("downstream_resolution_kind", resolutionKind)
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }



        private fun atomSubsetReceiptsFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            resolutionKind: String,
            witnessPattern: JSONObject,
            proofPayload: JSONObject
        ): JSONObject = JSONObject().apply {
            val subsetMode = witnessPattern.optString("subset_mode")
            val subsetSubtype = witnessPattern.optString("subset_subtype")
            val subsetHouse = witnessPattern.optJSONObject("house") ?: primaryHouse
            val subsetCells = witnessPattern.optJSONArray("subset_cells") ?: JSONArray()
            val lockedDigits = witnessPattern.optJSONArray("locked_digits") ?: JSONArray()
            val subsetCellCandidates = witnessPattern.optJSONArray("subset_cell_candidates") ?: JSONArray()

            val support = proofPayload.optJSONObject("support") ?: JSONObject()
            val cellOutcome = proofPayload.optJSONObject("cell_outcome") ?: JSONObject()
            val witnessByDigit = support.optJSONArray("witness_by_digit") ?: JSONArray()
            val blockerRowsIn = support.optJSONArray("blocker_rows") ?: JSONArray()
            val forcingSummary = support.optJSONObject("forcing_summary") ?: JSONObject()

            val blockedBySubsetDigits = JSONArray()
            val blockedByPeerDigits = JSONArray()
            val blockedByUnknownDigits = JSONArray()
            val blockerRows = JSONArray()
            val peerWitnessRows = JSONArray()

            fun relationHouseFor(
                witnessCellObj: JSONObject?,
                targetCell: CellRef,
                relation: String
            ): JSONObject? {
                val wIdx = witnessCellObj?.optInt("cellIndex", -1) ?: -1
                if (wIdx !in 0..80) return null

                val wr = (wIdx / 9) + 1
                val wc = (wIdx % 9) + 1

                return when (relation) {
                    "SAME_ROW" -> JSONObject().apply {
                        put("type", "row")
                        put("index1to9", targetCell.r)
                    }
                    "SAME_COL" -> JSONObject().apply {
                        put("type", "col")
                        put("index1to9", targetCell.c)
                    }
                    "SAME_BOX" -> JSONObject().apply {
                        put("type", "box")
                        put("index1to9", (((wr - 1) / 3) * 3 + ((wc - 1) / 3) + 1))
                    }
                    else -> null
                }
            }

            // Prefer already-normalized blocker_rows when present; otherwise reconstruct from witness_by_digit.
            if (blockerRowsIn.length() > 0) {
                for (i in 0 until blockerRowsIn.length()) {
                    val row = blockerRowsIn.optJSONObject(i) ?: continue
                    blockerRows.put(JSONObject(row.toString()))

                    val digit = row.optInt("digit", -1)
                    when (row.optString("source_kind")) {
                        "subset_witness" -> if (digit in 1..9) blockedBySubsetDigits.put(digit)
                        "peer_witness" -> {
                            if (digit in 1..9) blockedByPeerDigits.put(digit)

                            val witnessCell = row.optJSONObject("witness_cell")
                            val relation = row.optString("relation")
                            val relationHouse = relationHouseFor(witnessCell, focusCell, relation)

                            peerWitnessRows.put(JSONObject().apply {
                                put("digit", digit)
                                put("witness_cell", JSONObject(witnessCell?.toString() ?: "{}"))
                                put("relation", relation)
                                put("relation_house", JSONObject(relationHouse?.toString() ?: "{}"))
                            })
                        }
                        else -> if (digit in 1..9) blockedByUnknownDigits.put(digit)
                    }
                }
            } else {
                for (i in 0 until witnessByDigit.length()) {
                    val row = witnessByDigit.optJSONObject(i) ?: continue
                    val digit = row.optInt("digit", -1)
                    if (digit !in 1..9) continue

                    val witness = row.optJSONObject("witness") ?: JSONObject()
                    val kind = witness.optString("kind").ifBlank { "unknown" }

                    when (kind) {
                        "subset_group" -> {
                            blockedBySubsetDigits.put(digit)
                            blockerRows.put(JSONObject().apply {
                                put("digit", digit)
                                put("source_kind", "subset_witness")
                                put("subset_kind", witness.optString("subset_kind"))
                                put("subset_digits", JSONArray(witness.optJSONArray("digits")?.toString() ?: "[]"))
                                put("subset_cells", JSONArray(witness.optJSONArray("cells")?.toString() ?: "[]"))
                                put("house", JSONObject(witness.optJSONObject("house")?.toString() ?: "{}"))
                            })
                        }

                        "single_cell" -> {
                            blockedByPeerDigits.put(digit)

                            val witnessCell = witness.optJSONObject("cell")
                            val relation = witness.optString("relation")
                            val relationHouse = relationHouseFor(witnessCell, focusCell, relation)

                            blockerRows.put(JSONObject().apply {
                                put("digit", digit)
                                put("source_kind", "peer_witness")
                                put("witness_cell", JSONObject(witnessCell?.toString() ?: "{}"))
                                put("relation", relation)
                            })

                            peerWitnessRows.put(JSONObject().apply {
                                put("digit", digit)
                                put("witness_cell", JSONObject(witnessCell?.toString() ?: "{}"))
                                put("relation", relation)
                                put("relation_house", JSONObject(relationHouse?.toString() ?: "{}"))
                            })
                        }

                        else -> {
                            blockedByUnknownDigits.put(digit)
                            blockerRows.put(JSONObject().apply {
                                put("digit", digit)
                                put("source_kind", "unknown")
                            })
                        }
                    }
                }
            }

            val forcingHonestyNote = forcingSummary.optString("forcing_note").ifBlank {
                when {
                    blockedBySubsetDigits.length() > 0 && blockedByPeerDigits.length() > 0 ->
                        "The subset removes some candidates first, and the remaining row, column, or box blockers finish the collapse."
                    blockedBySubsetDigits.length() > 0 ->
                        "The subset itself performs the decisive cleanup on the target."
                    else ->
                        "The target collapses from the remaining blocker network around it."
                }
            }

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
            put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put("code", "SUBSET_TARGET_COLLAPSE")
                put("args", JSONObject().apply {
                    put("house", subsetHouse)
                    put("subset_mode", subsetMode)
                    put("subset_subtype", subsetSubtype)
                    put("subset_cells", JSONArray(subsetCells.toString()))
                    put("locked_digits", JSONArray(lockedDigits.toString()))
                    put("subset_cell_candidates", JSONArray(subsetCellCandidates.toString()))
                    put("target_cell", focusCell.toJson())
                    put("target_digit", focusDigit)
                    put("downstream_resolution_kind", resolutionKind)

                    put(
                        "claimed_candidate_digits",
                        JSONArray(cellOutcome.optJSONArray("claimed_candidate_digits")?.toString() ?: "[]")
                    )
                    put(
                        "remaining_candidate_digits",
                        JSONArray(cellOutcome.optJSONArray("remaining_candidate_digits")?.toString() ?: "[]")
                    )

                    put("blocked_by_subset_digits", blockedBySubsetDigits)
                    put("blocked_by_peer_digits", blockedByPeerDigits)
                    put("blocked_by_unknown_digits", blockedByUnknownDigits)
                    put("witness_by_digit", JSONArray(witnessByDigit.toString()))
                    put("blocker_rows", blockerRows)
                    put("peer_witness_rows", peerWitnessRows)

                    put("forcing_summary", JSONObject().apply {
                        put("technique_eliminated_digits", JSONArray(forcingSummary.optJSONArray("technique_eliminated_digits")?.toString() ?: "[]"))
                        put("peer_eliminated_digits", JSONArray(forcingSummary.optJSONArray("peer_eliminated_digits")?.toString() ?: "[]"))
                        put("unknown_eliminated_digits", JSONArray(forcingSummary.optJSONArray("unknown_eliminated_digits")?.toString() ?: "[]"))
                        put("remaining_candidate_digits", JSONArray(forcingSummary.optJSONArray("remaining_candidate_digits")?.toString() ?: "[]"))
                        put("forcing_note", forcingHonestyNote)
                    })
                })
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put("kind", NarrativeWitnessKindV1.OTHER.wire)
                    put("because", JSONObject().apply {
                        put("target_cell", focusCell.toJson())
                        put("target_digit", focusDigit)
                        put("blocked_by_subset_digits", JSONArray(blockedBySubsetDigits.toString()))
                        put("blocked_by_peer_digits", JSONArray(blockedByPeerDigits.toString()))
                        put("blocked_by_unknown_digits", JSONArray(blockedByUnknownDigits.toString()))
                        put("witness_by_digit", JSONArray(witnessByDigit.toString()))
                        put("blocker_rows", JSONArray(blockerRows.toString()))
                        put("peer_witness_rows", JSONArray(peerWitnessRows.toString()))
                        put("forcing_honesty_note", forcingHonestyNote)
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                put("frame_id", "ov:atom:$index")
                put("keep_subset_tableau", true)
                put("show_subset_house", true)
                put("show_subset_cells", true)
                put("show_subset_candidates", true)
                put("show_blocker_network", true)
                put("show_target_collapse", true)
                put("show_commit", false)
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
        }



        private fun atomLockInFromTruthV2(
            index: Int,
            focusCell: CellRef,
            focusDigit: Int,
            primaryHouse: JSONObject,
            resolutionKind: String,
            eliminatedPeers: List<Int>,
            eliminatedDigits: List<Int>,
            proofPayload: JSONObject
        ): JSONObject = JSONObject().apply {
            val isCellMode = resolutionKind == "CELL_CANDIDATE_DIGITS"
            val support = proofPayload.optJSONObject("support") ?: JSONObject()
            val witnessByDigit = support.optJSONArray("witness_by_digit") ?: JSONArray()

            val techniqueEliminatedDigits = JSONArray()
            val peerEliminatedDigits = JSONArray()
            val unknownEliminatedDigits = JSONArray()

            for (i in 0 until witnessByDigit.length()) {
                val row = witnessByDigit.optJSONObject(i) ?: continue
                val digit = row.optInt("digit", -1)
                if (digit !in 1..9) continue
                val witness = row.optJSONObject("witness") ?: JSONObject()
                when (witness.optString("kind")) {
                    "subset_group" -> techniqueEliminatedDigits.put(digit)
                    "single_cell" -> peerEliminatedDigits.put(digit)
                    else -> unknownEliminatedDigits.put(digit)
                }
            }

            val forcingHonestyNote = when {
                techniqueEliminatedDigits.length() > 0 && peerEliminatedDigits.length() > 0 ->
                    "The subset creates the opening, and the remaining blockers finish the job."
                techniqueEliminatedDigits.length() > 0 ->
                    "The subset cleanup is itself enough to force the target."
                else ->
                    "The remaining blockers around the target force the final candidate."
            }

            put("schema_version", "narrative_atom_v1")
            put("index", index)
            put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
            put("beat_kind", NarrativeBeatKindV1.LOCK_IN.wire)
            put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

            put("focus", JSONObject().apply {
                put("target_cell", focusCell.toJson())
                put("target_digit", focusDigit)
                put("primary_house", primaryHouse)
            })

            put("claim", JSONObject().apply {
                put(
                    "code",
                    if (isCellMode) {
                        NarrativeClaimCodeV1.ONLY_DIGIT_LEFT_FOR_CELL.wire
                    } else {
                        NarrativeClaimCodeV1.ONLY_CELL_LEFT_FOR_DIGIT_IN_HOUSE.wire
                    }
                )
                put("args", JSONObject().apply {
                    if (isCellMode) {
                        put("cell", focusCell.toJson())
                        put("remaining_digits", JSONArray().put(focusDigit))
                        put("eliminated_digits", JSONArray().apply { eliminatedDigits.forEach { put(it) } })
                        put("technique_eliminated_digits", JSONArray(techniqueEliminatedDigits.toString()))
                        put("peer_eliminated_digits", JSONArray(peerEliminatedDigits.toString()))
                        put("unknown_eliminated_digits", JSONArray(unknownEliminatedDigits.toString()))
                        put("forcing_honesty_note", forcingHonestyNote)
                    } else {
                        put("digit", focusDigit)
                        put("house", primaryHouse)
                        put("remaining_cells", JSONArray().put(focusCell.toJson()))
                        put("eliminated_peer_cell_indices", JSONArray().apply { eliminatedPeers.forEach { put(it) } })
                    }
                })
            })

            put("witnesses", JSONArray().put(
                JSONObject().apply {
                    put(
                        "kind",
                        if (isCellMode) {
                            NarrativeWitnessKindV1.CELL_REMAINS_ONE_DIGIT.wire
                        } else {
                            NarrativeWitnessKindV1.HOUSE_REMAINS_ONE.wire
                        }
                    )
                    put("because", JSONObject().apply {
                        put("remaining_cell", focusCell.toJson())
                        if (isCellMode) {
                            put("remaining_digit", focusDigit)
                            put("eliminated_digits", JSONArray().apply { eliminatedDigits.forEach { put(it) } })
                            put("technique_eliminated_digits", JSONArray(techniqueEliminatedDigits.toString()))
                            put("peer_eliminated_digits", JSONArray(peerEliminatedDigits.toString()))
                            put("unknown_eliminated_digits", JSONArray(unknownEliminatedDigits.toString()))
                            put("forcing_honesty_note", forcingHonestyNote)
                        } else {
                            put("eliminated_peer_cell_indices", JSONArray().apply { eliminatedPeers.forEach { put(it) } })
                        }
                    })
                }
            ))

            put("effects", JSONObject().apply {
                put("eliminations", JSONArray())
                put("placements", JSONArray())
            })

            put("overlay", JSONObject().apply {
                put("intent", OverlayIntentV1.SHOW_SWEEP.wire)
                put("frame_id", "ov:atom:$index")
            })
            put("user_prompt", JSONObject().apply { put("code", "ASK_USER_TRY") })
        }



        private fun atomTeachingNoteSubset(
            index: Int,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.SUBSETS.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.NONE.wire)

                put("focus", JSONObject().apply {
                    put("target_cell", cellRef(tr, tc, tIdx))
                    put("target_digit", td)
                    put("primary_house", primaryHouse)
                })

                put("claim", JSONObject().apply {
                    put("code", "TEACHING_NOTE_SUBSET")
                    put("args", JSONObject().apply {
                        put("note", "A subset locks a small set of digits into a small set of cells, or a small set of cells to a small set of digits.")
                    })
                })

                put("witnesses", JSONArray())
                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SPOTLIGHT.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        // ----------------------------
        // Fish
        // ----------------------------

        private fun atomFishPattern(index: Int, fish: FishData): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.FISH.wire)
                put("beat_kind", NarrativeBeatKindV1.LOCK_IN.wire)
                put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

                put("focus", JSONObject().apply {
                    put("target_cell", JSONObject().apply { put("r", 0); put("c", 0); put("cellIndex", -1) })
                    put("target_digit", fish.digit)
                    put("primary_house", houseRef(fish.baseType, fish.baseIndices.firstOrNull() ?: 1))
                })

                put("claim", JSONObject().apply {
                    put("code", NarrativeClaimCodeV1.FISH_LOCKS_DIGIT.wire)
                    put("args", JSONObject().apply {
                        put("fish_kind", fish.fishKind)
                        put("digit", fish.digit)
                        put("base", JSONObject().apply {
                            put("type", fish.baseType)
                            put("indices", JSONArray().apply { fish.baseIndices.forEach { put(it) } })
                        })
                        put("cover", JSONObject().apply {
                            put("type", fish.coverType)
                            put("indices", JSONArray().apply { fish.coverIndices.forEach { put(it) } })
                        })
                        put("corners", JSONArray().apply { fish.corners.forEach { put(it) } })
                    })
                })

                put("witnesses", JSONArray().put(
                    JSONObject().apply {
                        put("kind", NarrativeWitnessKindV1.OTHER.wire)
                        put("because", JSONObject().apply {
                            put("fish_kind", fish.fishKind)
                            put("corners", JSONArray().apply { fish.corners.forEach { put(it) } })
                        })
                    }
                ))

                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SWEEP.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        private fun atomFishSweep(index: Int, fish: FishData): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.FISH.wire)
                put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
                put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

                put("focus", JSONObject().apply {
                    put("target_cell", JSONObject().apply { put("r", 0); put("c", 0); put("cellIndex", -1) })
                    put("target_digit", fish.digit)
                    put("primary_house", houseRef(fish.coverType, fish.coverIndices.firstOrNull() ?: 1))
                })

                put("claim", JSONObject().apply {
                    put("code", "FISH_SWEEP")
                    put("args", JSONObject().apply {
                        put("fish_kind", fish.fishKind)
                        put("digit", fish.digit)
                        put("sweep_cells", JSONArray().apply { fish.sweepCells.forEach { put(it) } })
                    })
                })

                put("witnesses", JSONArray().put(
                    JSONObject().apply {
                        put("kind", NarrativeWitnessKindV1.OTHER.wire)
                        put("because", JSONObject().apply {
                            put("fish_corners", JSONArray().apply { fish.corners.forEach { put(it) } })
                            put("sweep_cells", JSONArray().apply { fish.sweepCells.forEach { put(it) } })
                        })
                    }
                ))

                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray().apply {
                        fish.sweepCells.forEach { idx ->
                            put(JSONObject().apply {
                                put("cellIndex", idx)
                                put("digit", fish.digit)
                                put("reason_code", "fish_sweep")
                            })
                        }
                    })
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        private fun atomTeachingNoteFish(
            index: Int,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.FISH.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.NONE.wire)
                put("focus", JSONObject().apply {
                    put("target_cell", cellRef(tr, tc, tIdx))
                    put("target_digit", td)
                    put("primary_house", primaryHouse)
                })
                put("claim", JSONObject().apply {
                    put("code", "TEACHING_NOTE_FISH")
                    put("args", JSONObject().apply {
                        put("note", "A fish locks a digit into matching rows and columns, so other candidates in those cover lines get removed.")
                    })
                })
                put("witnesses", JSONArray())
                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })
                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SPOTLIGHT.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        // ----------------------------
        // Wings
        // ----------------------------

        private fun atomWingEitherWay(index: Int, wing: WingData): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.WINGS.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

                val hr = (wing.hinge / 9) + 1
                val hc = (wing.hinge % 9) + 1

                put("focus", JSONObject().apply {
                    put("target_cell", JSONObject().apply {
                        put("r", hr)
                        put("c", hc)
                        put("cellIndex", wing.hinge)
                    })
                    put("target_digit", wing.digit)
                    put("primary_house", houseRef("box", boxIndex(hr, hc)))
                })

                put("claim", JSONObject().apply {
                    put("code", NarrativeClaimCodeV1.EITHER_WAY_ELIMINATION.wire)
                    put("args", JSONObject().apply {
                        put("digit", wing.digit)
                        put("hinge", wing.hinge)
                        put("pincers", JSONArray().apply { wing.pincers.forEach { put(it) } })
                        put("target_eliminate", wing.targetEliminate)
                    })
                })

                put("witnesses", JSONArray().put(
                    JSONObject().apply {
                        put("kind", NarrativeWitnessKindV1.SEES_BOTH.wire)
                        put("because", JSONObject().apply {
                            put("hinge", wing.hinge)
                            put("pincers", JSONArray().apply { wing.pincers.forEach { put(it) } })
                        })
                    }
                ))

                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SWEEP.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        private fun atomWingElimination(index: Int, wing: WingData): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.WINGS.wire)
                put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
                put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

                val tr = (wing.targetEliminate / 9) + 1
                val tc = (wing.targetEliminate % 9) + 1

                put("focus", JSONObject().apply {
                    put("target_cell", JSONObject().apply {
                        put("r", tr)
                        put("c", tc)
                        put("cellIndex", wing.targetEliminate)
                    })
                    put("target_digit", wing.digit)
                    put("primary_house", houseRef("row", tr))
                })

                put("claim", JSONObject().apply {
                    put("code", NarrativeClaimCodeV1.CELL_CANNOT_BE_DIGIT.wire)
                    put("args", JSONObject().apply {
                        put("cell", cellRef(tr, tc, wing.targetEliminate))
                        put("digit", wing.digit)
                    })
                })

                put("witnesses", JSONArray().put(
                    JSONObject().apply {
                        put("kind", NarrativeWitnessKindV1.SEES_BOTH.wire)
                        put("because", JSONObject().apply {
                            put("hinge", wing.hinge)
                            put("pincers", JSONArray().apply { wing.pincers.forEach { put(it) } })
                            put("target", wing.targetEliminate)
                        })
                    }
                ))

                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray().put(
                        JSONObject().apply {
                            put("cellIndex", wing.targetEliminate)
                            put("digit", wing.digit)
                            put("reason_code", "wing_either_way")
                        }
                    ))
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        private fun atomTeachingNoteWing(
            index: Int,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.WINGS.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.NONE.wire)
                put("focus", JSONObject().apply {
                    put("target_cell", cellRef(tr, tc, tIdx))
                    put("target_digit", td)
                    put("primary_house", primaryHouse)
                })
                put("claim", JSONObject().apply {
                    put("code", "TEACHING_NOTE_WING")
                    put("args", JSONObject().apply {
                        put("note", "A wing means whichever branch you take, the same candidate gets eliminated elsewhere.")
                    })
                })
                put("witnesses", JSONArray())
                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })
                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SPOTLIGHT.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        // ----------------------------
        // Chains
        // ----------------------------

        private fun atomChainColoring(index: Int, chain: ChainData): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.CHAINS.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

                val firstIdx = chain.colorA.firstOrNull() ?: -1
                val fr = if (firstIdx in 0..80) (firstIdx / 9) + 1 else 1

                put("focus", JSONObject().apply {
                    put("target_cell", JSONObject().apply {
                        put("r", if (firstIdx in 0..80) (firstIdx / 9) + 1 else 0)
                        put("c", if (firstIdx in 0..80) (firstIdx % 9) + 1 else 0)
                        put("cellIndex", firstIdx)
                    })
                    put("target_digit", chain.digit)
                    put("primary_house", houseRef("row", fr))
                })

                put("claim", JSONObject().apply {
                    put("code", "CHAIN_COLORING")
                    put("args", JSONObject().apply {
                        put("digit", chain.digit)
                        put("colorA", JSONArray().apply { chain.colorA.forEach { put(it) } })
                        put("colorB", JSONArray().apply { chain.colorB.forEach { put(it) } })
                        put("eliminate_cell", chain.eliminateCell ?: JSONObject.NULL)
                    })
                })

                put("witnesses", JSONArray().put(
                    JSONObject().apply {
                        put("kind", NarrativeWitnessKindV1.CHAIN_CONTRADICTION.wire)
                        put("because", JSONObject().apply {
                            put("colorA", JSONArray().apply { chain.colorA.forEach { put(it) } })
                            put("colorB", JSONArray().apply { chain.colorB.forEach { put(it) } })
                        })
                    }
                ))

                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SWEEP.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        private fun atomChainContradiction(index: Int, chain: ChainData): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.CHAINS.wire)
                put("beat_kind", NarrativeBeatKindV1.WITNESS_ELIMINATION.wire)
                put("spoiler_level", SpoilerLevelV1.CANDIDATE.wire)

                val elimIdx = chain.eliminateCell ?: -1
                val er = if (elimIdx in 0..80) (elimIdx / 9) + 1 else 1
                val ec = if (elimIdx in 0..80) (elimIdx % 9) + 1 else 1

                put("focus", JSONObject().apply {
                    put("target_cell", JSONObject().apply {
                        put("r", er)
                        put("c", ec)
                        put("cellIndex", elimIdx)
                    })
                    put("target_digit", chain.digit)
                    put("primary_house", houseRef("row", er))
                })

                put("claim", JSONObject().apply {
                    put("code", NarrativeClaimCodeV1.CONTRADICTION_IMPLES_NOT.wire)
                    put("args", JSONObject().apply {
                        put("digit", chain.digit)
                        put("eliminate_cell", chain.eliminateCell ?: JSONObject.NULL)
                    })
                })

                put("witnesses", JSONArray().put(
                    JSONObject().apply {
                        put("kind", NarrativeWitnessKindV1.CHAIN_CONTRADICTION.wire)
                        put("because", JSONObject().apply {
                            put("colorA", JSONArray().apply { chain.colorA.forEach { put(it) } })
                            put("colorB", JSONArray().apply { chain.colorB.forEach { put(it) } })
                            put("eliminate_cell", chain.eliminateCell ?: JSONObject.NULL)
                        })
                    }
                ))

                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray().apply {
                        val idx = chain.eliminateCell
                        if (idx != null && idx in 0..80) {
                            put(JSONObject().apply {
                                put("cellIndex", idx)
                                put("digit", chain.digit)
                                put("reason_code", "chain_contradiction")
                            })
                        }
                    })
                    put("placements", JSONArray())
                })

                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_WITNESS.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        private fun atomTeachingNoteChain(
            index: Int,
            tr: Int,
            tc: Int,
            tIdx: Int,
            td: Int,
            primaryHouse: JSONObject
        ): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atom_v1")
                put("index", index)
                put("archetype", NarrativeArchetypeV1.CHAINS.wire)
                put("beat_kind", NarrativeBeatKindV1.TEACHING_NOTE.wire)
                put("spoiler_level", SpoilerLevelV1.NONE.wire)
                put("focus", JSONObject().apply {
                    put("target_cell", cellRef(tr, tc, tIdx))
                    put("target_digit", td)
                    put("primary_house", primaryHouse)
                })
                put("claim", JSONObject().apply {
                    put("code", "TEACHING_NOTE_CHAIN")
                    put("args", JSONObject().apply {
                        put("note", "A chain alternates possibilities. If one branch breaks, a candidate can be safely removed.")
                    })
                })
                put("witnesses", JSONArray())
                put("effects", JSONObject().apply {
                    put("eliminations", JSONArray())
                    put("placements", JSONArray())
                })
                put("overlay", JSONObject().apply {
                    put("intent", OverlayIntentV1.SHOW_SPOTLIGHT.wire)
                    put("frame_id", "ov:atom:$index")
                })
                put("user_prompt", JSONObject().apply { put("code", "ASK_NEXT_HINT") })
            }

        // ----------------------------
        // Basic helpers
        // ----------------------------

        private fun emptyPacket(reason: String): JSONObject =
            JSONObject().apply {
                put("schema_version", "narrative_atoms_v1")
                put("archetype", NarrativeArchetypeV1.UNKNOWN.wire)
                put("atoms", JSONArray())
                put("version_note", "empty:$reason")
            }

        private fun cellRef(r: Int, c: Int, idx: Int): JSONObject =
            JSONObject().apply {
                put("r", r)
                put("c", c)
                put("cellIndex", idx)
            }

        private fun houseRef(type: String, index1to9: Int): JSONObject =
            JSONObject().apply {
                put("type", type)
                put("index1to9", index1to9)
            }

        private fun boxIndex(r: Int, c: Int): Int =
            ((r - 1) / 3) * 3 + ((c - 1) / 3) + 1

        private fun relationBetweenCells(aIdx: Int, bIdx: Int): String {
            val ar = (aIdx / 9) + 1
            val ac = (aIdx % 9) + 1
            val br = (bIdx / 9) + 1
            val bc = (bIdx % 9) + 1
            return when {
                ar == br -> "SAME_ROW"
                ac == bc -> "SAME_COL"
                boxIndex(ar, ac) == boxIndex(br, bc) -> "SAME_BOX"
                else -> "RELATED"
            }
        }

        private fun houseCellIndicesOrEmpty(house: JSONObject?): List<Int> {
            val type = house?.optString("type").orEmpty()
            val idx = house?.optInt("index1to9", -1) ?: -1
            return when (type) {
                "row" -> if (idx in 1..9) (0 until 9).map { c -> (idx - 1) * 9 + c } else emptyList()
                "col" -> if (idx in 1..9) (0 until 9).map { r -> r * 9 + (idx - 1) } else emptyList()
                "box" -> {
                    if (idx !in 1..9) emptyList()
                    else {
                        val br = ((idx - 1) / 3) * 3
                        val bc = ((idx - 1) % 3) * 3
                        buildList {
                            for (dr in 0 until 3) {
                                for (dc in 0 until 3) {
                                    add((br + dr) * 9 + (bc + dc))
                                }
                            }
                        }
                    }
                }
                else -> emptyList()
            }
        }

        private fun parseCellRef(raw: Any?): CellRef? {
            return when (raw) {
                is JSONObject -> {
                    if (raw.has("cell")) {
                        parseCellRef(raw.opt("cell"))
                    } else {
                        val idx = raw.optInt("cellIndex", raw.optInt("cell_index", -1))
                        val r = raw.optInt("r", if (idx in 0..80) (idx / 9) + 1 else -1)
                        val c = raw.optInt("c", if (idx in 0..80) (idx % 9) + 1 else -1)
                        if (idx in 0..80 && r in 1..9 && c in 1..9) CellRef(r, c, idx) else null
                    }
                }
                is Int -> if (raw in 0..80) CellRef((raw / 9) + 1, (raw % 9) + 1, raw) else null
                else -> null
            }
        }

        private fun parseCellList(arr: JSONArray?): List<CellRef> {
            if (arr == null) return emptyList()
            val out = mutableListOf<CellRef>()
            for (i in 0 until arr.length()) {
                parseCellRef(arr.opt(i))?.let { out += it }
            }
            return out
        }



        private fun firstCell(arr: JSONArray?): CellRef? =
            parseCellList(arr).firstOrNull()

        private fun parseHouse(raw: Any?): JSONObject? {
            if (raw !is JSONObject) return null
            val type = raw.optString("type")
            val idx = raw.optInt("index1to9", -1)
            return if (type in setOf("row", "col", "box", "region", "cell") && (idx in 1..9 || type == "cell")) {
                JSONObject(raw.toString())
            } else {
                null
            }
        }

        private fun firstHouse(arr: JSONArray?): JSONObject? {
            if (arr == null) return null
            for (i in 0 until arr.length()) {
                parseHouse(arr.opt(i))?.let { return it }
            }
            return null
        }

        private fun parseHouseList(arr: JSONArray?): List<JSONObject> {
            if (arr == null) return emptyList()
            val out = mutableListOf<JSONObject>()
            for (i in 0 until arr.length()) {
                parseHouse(arr.opt(i))?.let { out += it }
            }
            return out
        }

        private fun parseIntList(arr: JSONArray?): List<Int> {
            if (arr == null) return emptyList()
            val out = mutableListOf<Int>()
            for (i in 0 until arr.length()) {
                val n = arr.optInt(i, Int.MIN_VALUE)
                if (n != Int.MIN_VALUE) out += n
            }
            return out
        }

        private fun firstPlacement(arr: JSONArray?): Placement? =
            parsePlacements(arr).firstOrNull()

        private fun parsePlacements(arr: JSONArray?): List<Placement> {
            if (arr == null) return emptyList()
            val out = mutableListOf<Placement>()
            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                val cell = parseCellRef(obj.opt("cell")) ?: parseCellRef(obj)
                val digit = obj.optInt("digit", -1)
                if (cell != null && digit in 1..9) {
                    out += Placement(cell, digit)
                }
            }
            return out
        }

        private fun parseCandidateElims(arr: JSONArray?): List<CandidateElim> {
            if (arr == null) return emptyList()
            val out = mutableListOf<CandidateElim>()
            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                val cell = parseCellRef(obj.opt("cell")) ?: parseCellRef(obj)
                val digit = obj.optInt("digit", -1)
                if (cell != null && digit in 1..9) {
                    out += CandidateElim(cell, digit)
                }
            }
            return out
        }

        private fun parseCandidateRestrictions(arr: JSONArray?): List<CandidateRestriction> {
            if (arr == null) return emptyList()
            val out = mutableListOf<CandidateRestriction>()
            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                val cell = parseCellRef(obj.opt("cell")) ?: parseCellRef(obj)
                if (cell == null) continue
                val removed = parseIntList(obj.optJSONArray("removed_digits")).filter { it in 1..9 }.distinct().sorted()
                val remaining = parseIntList(obj.optJSONArray("remaining_digits")).filter { it in 1..9 }.distinct().sorted()
                out += CandidateRestriction(cell, removed, remaining)
            }
            return out
        }

        // ----------------------------
        // Narrative Truth V2 helpers
        // ----------------------------

        private fun cellHouseRef(cellIndex: Int): JSONObject {
            val r = (cellIndex / 9) + 1
            val c = (cellIndex % 9) + 1
            return JSONObject().apply {
                put("type", "cell")
                put("cell", cellRef(r, c, cellIndex))
            }
        }

        private fun buildDownstreamResolutionV2(cell: CellRef, digit: Int): JSONObject =
            JSONObject().apply {
                put("placement", JSONObject().apply {
                    put("cell", cell.toJson())
                    put("digit", digit)
                })
            }


        private fun buildFinalResolutionContractV2(
            kind: String,
            primaryHouse: JSONObject?,
            focusCell: CellRef?,
            digit: Int?,
            originStory: JSONObject? = null
        ): JSONObject =
            JSONObject().apply {
                put("kind", kind)
                put("primary_house", primaryHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                put("focus_cell", focusCell?.toJson() ?: JSONObject.NULL)
                put("digit", digit?.takeIf { it in 1..9 } ?: JSONObject.NULL)
                put("origin_story", originStory?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
            }




        private fun buildHiddenSingleProofPayloadV2(hs: HiddenSingleCanonical): JSONObject {
            val peerCells = hs.peerWitnessPairs.map { it.first }.distinctBy { it.cellIndex }
            val witnessByCell = JSONArray().apply {
                hs.peerWitnessPairs.forEach { (peer, witness) ->
                    put(JSONObject().apply {
                        put("claimed_cell", peer.toJson())
                        put("witness", JSONObject().apply {
                            put("kind", "single_cell")
                            put("digit", hs.digit)
                            put("cell", witness.toJson())
                            put("relation", relationBetweenCells(peer.cellIndex, witness.cellIndex))
                        })
                    })
                }
            }

            return JSONObject().apply {
                put("digit", hs.digit)
                put("house_claim", JSONObject().apply {
                    put("digit", hs.digit)
                    put("house", JSONObject(hs.primaryHouse.toString()))
                    put("default_candidate_cells", JSONArray().apply {
                        peerCells.forEach { put(it.toJson()) }
                        put(cellRef(hs.focusR, hs.focusC, hs.focusIdx))
                    })
                    put("claimed_candidate_cells", JSONArray().apply {
                        peerCells.forEach { put(it.toJson()) }
                    })
                    put("remaining_candidate_cells", JSONArray().put(
                        cellRef(hs.focusR, hs.focusC, hs.focusIdx)
                    ))
                })
                put("support", JSONObject().apply {
                    put("peer_cells", JSONArray().apply { peerCells.forEach { put(it.toJson()) } })
                    put("witness_by_cell", witnessByCell)
                })
            }
        }

        private fun buildNakedSingleProofPayloadV2(ns: NakedSingleCanonical): JSONObject {
            val witnessCells = ns.digitWitnessPairs.map { it.second }.distinctBy { it.cellIndex }
            return JSONObject().apply {
                put("cell_outcome", JSONObject().apply {
                    put("cell", cellRef(ns.focusR, ns.focusC, ns.focusIdx))
                    put("default_candidate_digits", JSONArray().apply { ns.defaultCandidateDigits.forEach { put(it) } })
                    put("claimed_candidate_digits", JSONArray().apply { ns.eliminatedDigits.forEach { put(it) } })
                    put("remaining_candidate_digits", JSONArray().put(ns.digit))
                })
                put("support", JSONObject().apply {
                    put("witness_cells", JSONArray().apply { witnessCells.forEach { put(it.toJson()) } })
                    put("witness_by_digit", JSONArray().apply {
                        ns.digitWitnessPairs.forEach { (digit, witness) ->
                            put(JSONObject().apply {
                                put("digit", digit)
                                put("witness", JSONObject().apply {
                                    put("kind", "single_cell")
                                    put("cell", witness.toJson())
                                    put("relation", relationBetweenCells(ns.focusIdx, witness.cellIndex))
                                })
                            })
                        }
                    })
                    put("eliminated_digits", JSONArray().apply { ns.eliminatedDigits.forEach { put(it) } })
                })
            }
        }

        private fun buildFullHouseProofPayloadV2(fh: FullHouseCanonical): JSONObject =
            JSONObject().apply {
                put("digit", fh.digit)
                put("house_claim", JSONObject().apply {
                    put("digit", fh.digit)
                    put("house", JSONObject(fh.primaryHouse.toString()))
                    put("remaining_candidate_cells", JSONArray().put(fh.remainingCell.toJson()))
                    put("remaining_candidate_digits", JSONArray().put(fh.remainingDigit))
                    put("filled_digits", JSONArray().apply { fh.filledDigits.forEach { put(it) } })
                })
                put("support", JSONObject().apply {
                    put("remaining_cell", fh.remainingCell.toJson())
                    put("remaining_digit", fh.remainingDigit)
                    put("filled_digits", JSONArray().apply { fh.filledDigits.forEach { put(it) } })
                })
            }

        private fun buildIntersectionDownstreamResolutionV2(ci: CanonicalIntersection): JSONObject {
            val cell = ci.focusCell
            val digit = ci.targetDigit
            val kind = ci.downstreamResolutionKind

            return if (
                kind == "CELL_CANDIDATE_DIGITS" &&
                cell != null &&
                digit != null &&
                digit in 1..9
            ) {
                buildDownstreamResolutionV2(cell, digit)
            } else {
                JSONObject()
            }
        }

        private fun buildIntersectionProofPayloadV2(ci: CanonicalIntersection): JSONObject {
            val sourceConfinementProof = ci.sourceConfinementProof?.let { JSONObject(it.toString()) } ?: JSONObject()

            val overlapCells =
                sourceConfinementProof.optJSONArray("overlap_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray().apply { ci.lockedCells.forEach { put(it.toJson()) } }

            val sourceOutsideOverlapCells =
                sourceConfinementProof.optJSONArray("source_outside_overlap_cells") ?: JSONArray()

            val crossOutsideOverlapCells =
                sourceConfinementProof.optJSONArray("cross_outside_overlap_cells") ?: JSONArray()

            val forbiddenCrossCells =
                sourceConfinementProof.optJSONArray("forbidden_cross_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray().apply { ci.sweepCells.forEach { put(it.toJson()) } }

            return JSONObject().apply {
                put("intersection_claim", JSONObject().apply {
                    put("digit", ci.digit ?: JSONObject.NULL)
                    put("interaction_kind", ci.interactionKind)
                    put("source_house", JSONObject(ci.sourceHouse.toString()))
                    put("cross_house", JSONObject(ci.targetHouse.toString()))
                    put("target_house", JSONObject(ci.targetHouse.toString())) // backward-compatible alias
                    put("box_house", ci.boxHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                    put("line_house", ci.lineHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                    put("line_type", ci.lineType ?: JSONObject.NULL)
                    put("orientation", ci.orientation ?: JSONObject.NULL)

                    put("overlap_cells", JSONArray(overlapCells.toString()))
                    put("pattern_cells", JSONArray(overlapCells.toString()))
                    put("locked_cells", JSONArray(overlapCells.toString())) // backward-compatible alias
                    put("source_outside_overlap_cells", JSONArray(sourceOutsideOverlapCells.toString()))
                    put("cross_outside_overlap_cells", JSONArray(crossOutsideOverlapCells.toString()))
                    put("forbidden_cross_cells", JSONArray(forbiddenCrossCells.toString()))
                    put("sweep_cells", JSONArray().apply { ci.sweepCells.forEach { put(it.toJson()) } })
                    put("cardinality", overlapCells.length())
                    put(
                        "pattern_subtype",
                        "${ci.interactionKind.lowercase()}_" +
                                when (overlapCells.length()) {
                                    2 -> "pair"
                                    3 -> "triple"
                                    else -> "group"
                                }
                    )
                })

                put("support", JSONObject().apply {
                    put("witness_cells", JSONArray().apply {
                        ci.witnessCells.forEach { put(it.toJson()) }
                    })
                    put("explanation_links", JSONArray().apply {
                        ci.explanationLinks.forEach { row ->
                            put(JSONObject().apply {
                                put("kind", row.kind)
                                put("digit", row.digit ?: JSONObject.NULL)
                                put("interaction_kind", row.interactionKind ?: JSONObject.NULL)
                                put("locked_cells", JSONArray().apply {
                                    row.lockedCells.forEach { put(it.toJson()) }
                                })
                                put("sweep_cell", row.sweepCell?.toJson() ?: JSONObject.NULL)
                                put("source_house", row.sourceHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                                put("target_house", row.targetHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                                put("box_house", row.boxHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                                put("line_house", row.lineHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                                put("line_type", row.lineType ?: JSONObject.NULL)
                                put("orientation", row.orientation ?: JSONObject.NULL)
                            })
                        }
                    })
                    put(
                        "source_confinement_proof",
                        if (sourceConfinementProof.length() > 0) JSONObject(sourceConfinementProof.toString()) else JSONObject.NULL
                    )
                    put(
                        "final_canonical_proof",
                        ci.finalCanonicalProof?.let { JSONObject(it.toString()) } ?: JSONObject.NULL
                    )
                })

                put(
                    "final_canonical_proof",
                    ci.finalCanonicalProof?.let { JSONObject(it.toString()) } ?: JSONObject.NULL
                )

                put("eliminations", JSONArray().apply {
                    ci.immediateEliminations.forEach { elim ->
                        put(JSONObject().apply {
                            put("cell", elim.cell.toJson())
                            put("digit", elim.digit)
                        })
                    }
                })
            }
        }

        private fun buildIntersectionWitnessPatternV2(ci: CanonicalIntersection): JSONObject {
            val sourceConfinementProof = ci.sourceConfinementProof?.let { JSONObject(it.toString()) } ?: JSONObject()

            val overlapCells =
                sourceConfinementProof.optJSONArray("overlap_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray().apply { ci.lockedCells.forEach { put(it.toJson()) } }

            val sourceOutsideOverlapCells =
                sourceConfinementProof.optJSONArray("source_outside_overlap_cells") ?: JSONArray()

            val crossOutsideOverlapCells =
                sourceConfinementProof.optJSONArray("cross_outside_overlap_cells") ?: JSONArray()

            val forbiddenCrossCells =
                sourceConfinementProof.optJSONArray("forbidden_cross_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray().apply { ci.sweepCells.forEach { put(it.toJson()) } }

            val kind = when (ci.interactionKind.lowercase()) {
                "pointing" -> "INTERSECTION_POINTING"
                "claiming" -> "INTERSECTION_CLAIMING"
                else -> "INTERSECTION"
            }

            return JSONObject().apply {
                put("kind", kind)
                put("interaction_kind", ci.interactionKind)
                put("direction_mode", ci.interactionKind)
                put("digit", ci.digit ?: JSONObject.NULL)
                put("source_house", JSONObject(ci.sourceHouse.toString()))
                put("cross_house", JSONObject(ci.targetHouse.toString()))
                put("target_house", JSONObject(ci.targetHouse.toString())) // backward-compatible alias
                put("box_house", ci.boxHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                put("line_house", ci.lineHouse?.let { JSONObject(it.toString()) } ?: JSONObject.NULL)
                put("line_type", ci.lineType ?: JSONObject.NULL)
                put("orientation", ci.orientation ?: JSONObject.NULL)
                put(
                    "pattern_subtype",
                    "${ci.interactionKind.lowercase()}_" +
                            when (overlapCells.length()) {
                                2 -> "pair"
                                3 -> "triple"
                                else -> "group"
                            }
                )
                put("cardinality", overlapCells.length())

                put("overlap_cells", JSONArray(overlapCells.toString()))
                put("pattern_cells", JSONArray(overlapCells.toString()))
                put("locked_cells", JSONArray(overlapCells.toString())) // backward-compatible alias
                put("source_outside_overlap_cells", JSONArray(sourceOutsideOverlapCells.toString()))
                put("cross_outside_overlap_cells", JSONArray(crossOutsideOverlapCells.toString()))
                put("forbidden_cross_cells", JSONArray(forbiddenCrossCells.toString()))
                put("sweep_cells", JSONArray().apply {
                    ci.sweepCells.forEach { put(it.toJson()) }
                })
            }
        }

        private fun buildIntersectionTriggerPatternV2(ci: CanonicalIntersection): JSONObject =
            JSONObject(buildIntersectionWitnessPatternV2(ci).toString())

        private fun buildIntersectionTriggerExplanationV2(
            ci: CanonicalIntersection,
            proofPayload: JSONObject
        ): JSONObject {
            val support = proofPayload.optJSONObject("support") ?: JSONObject()
            val sourceConfinementProof = support.optJSONObject("source_confinement_proof") ?: JSONObject()

            val overlapSurvivorCells =
                sourceConfinementProof.optJSONArray("surviving_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("overlap_cells")
                    ?: JSONArray()

            val outsideOpenSeatAuditRows =
                sourceConfinementProof.optJSONArray("outside_open_seat_audit_rows")
                    ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("outside_audit_rows")
                        ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("eliminated_source_cells")
                    ?: JSONArray()

            val witnessClosureRows =
                sourceConfinementProof.optJSONArray("outside_witness_closure_rows")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray()

            val openNoncandidateRows =
                sourceConfinementProof.optJSONArray("outside_open_noncandidate_rows")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray()

            val setupPreferredAuditRows =
                sourceConfinementProof.optJSONArray("setup_preferred_audit_rows")
                    ?.takeIf { it.length() > 0 }
                    ?: witnessClosureRows
                        .takeIf { it.length() > 0 }
                    ?: outsideOpenSeatAuditRows

            val outsideOpenSeatCells =
                sourceConfinementProof.optJSONArray("source_outside_overlap_open_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: sourceConfinementProof.optJSONArray("source_outside_overlap_cells")
                    ?: JSONArray()

            val outsideOpenCandidateCells =
                sourceConfinementProof.optJSONArray("source_outside_overlap_open_candidate_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray()

            val outsideOpenNoncandidateCells =
                sourceConfinementProof.optJSONArray("source_outside_overlap_open_noncandidate_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray()

            val subtype =
                "${ci.interactionKind.lowercase()}_" +
                        when ((sourceConfinementProof.optJSONArray("overlap_cells") ?: JSONArray()).length().takeIf { it > 0 }
                            ?: ci.lockedCells.size) {
                            2 -> "pair"
                            3 -> "triple"
                            else -> "group"
                        }

            return JSONObject().apply {
                put("kind", "INTERSECTION_TRIGGER_EXPLANATION")
                put("interaction_kind", ci.interactionKind)
                put("digit", ci.digit ?: JSONObject.NULL)
                put("semantic_style", "source_confinement")

                put("source_house_outside_overlap_audit", JSONArray(outsideOpenSeatAuditRows.toString()))
                put("source_house_outside_open_seat_audit", JSONArray(outsideOpenSeatAuditRows.toString()))
                put("source_house_witness_closure_rows", JSONArray(witnessClosureRows.toString()))
                put("source_house_open_noncandidate_rows", JSONArray(openNoncandidateRows.toString()))
                put("setup_preferred_audit_rows", JSONArray(setupPreferredAuditRows.toString()))

                put("outside_open_seat_cells", JSONArray(outsideOpenSeatCells.toString()))
                put("outside_open_candidate_cells", JSONArray(outsideOpenCandidateCells.toString()))
                put("outside_open_noncandidate_cells", JSONArray(outsideOpenNoncandidateCells.toString()))

                put("overlap_survivor_cells", JSONArray(overlapSurvivorCells.toString()))
                put(
                    "forced_into_overlap_summary",
                    sourceConfinementProof.opt("forced_into_overlap_summary")
                        ?: sourceConfinementProof.opt("forced_inward_reason")
                        ?: sourceConfinementProof.optString("conclusion")
                            .takeIf { it.isNotBlank() }
                        ?: "Among the open seats outside the overlap, the source house has nowhere left to place the digit."
                )
                put(
                    "forced_inward_reason",
                    sourceConfinementProof.opt("forced_inward_reason")
                        ?: sourceConfinementProof.opt("forced_into_overlap_summary")
                        ?: "Among the open seats outside the overlap, the source house has nowhere left to place the digit."
                )
                put(
                    "pattern_reveal_moment",
                    sourceConfinementProof.opt("pattern_reveal_moment")
                        ?: "The remaining overlap cells become the forced carriers of the digit."
                )
                put(
                    "source_confinement_proof",
                    if (sourceConfinementProof.length() > 0) {
                        JSONObject(sourceConfinementProof.toString())
                    } else {
                        JSONObject.NULL
                    }
                )
                put("pattern_subtype", subtype)
            }
        }

        private fun buildIntersectionTriggerBridgeV2(ci: CanonicalIntersection): JSONObject {
            val sourceConfinementProof = ci.sourceConfinementProof?.let { JSONObject(it.toString()) } ?: JSONObject()
            val forbiddenCrossCells =
                sourceConfinementProof.optJSONArray("forbidden_cross_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray().apply { ci.sweepCells.forEach { put(it.toJson()) } }

            return JSONObject().apply {
                put("kind", "INTERSECTION_TRIGGER_BRIDGE")
                put("interaction_kind", ci.interactionKind)
                put("direction_mode", ci.interactionKind)
                put("final_resolution_kind", ci.finalResolutionKind)
                put("downstream_resolution_kind", ci.finalResolutionKind)
                put("source_house", JSONObject(ci.sourceHouse.toString()))
                put("cross_house", JSONObject(ci.targetHouse.toString()))
                put("target_house", JSONObject(ci.targetHouse.toString())) // backward-compatible alias
                put("overlap_cells", sourceConfinementProof.optJSONArray("overlap_cells") ?: JSONArray().apply {
                    ci.lockedCells.forEach { put(it.toJson()) }
                })
                put("forbidden_elsewhere_cells", JSONArray(forbiddenCrossCells.toString()))
                put("cross_house_now_restricted", true)
                put(
                    "cross_house_permission_change",
                    when (ci.finalResolutionKind) {
                        "CELL_CANDIDATE_DIGITS" ->
                            "Once the digit is trapped in the overlap, the crossing house must surrender it elsewhere, which helps collapse the downstream target."
                        "HOUSE_CANDIDATE_CELLS_FOR_DIGIT" ->
                            "Once the digit is trapped in the overlap, the crossing house must surrender it elsewhere, which narrows the remaining legal seats."
                        else ->
                            "Once the digit is trapped in the overlap, the crossing house is no longer free to keep it elsewhere."
                    }
                )
                put(
                    "why_this_matters",
                    when (ci.finalResolutionKind) {
                        "CELL_CANDIDATE_DIGITS" ->
                            "The overlap trap redraws what the crossing house is allowed to keep, preparing the downstream survivor."
                        "HOUSE_CANDIDATE_CELLS_FOR_DIGIT" ->
                            "The overlap trap redraws where the digit may still sit in the affected house."
                        else ->
                            "The overlap trap changes what remains legal around the target."
                    }
                )
            }
        }

        private fun buildIntersectionTriggerPacketV2(
            ci: CanonicalIntersection,
            proofPayload: JSONObject
        ): JSONObject {
            val triggerPattern = buildIntersectionTriggerPatternV2(ci)
            val triggerExplanation = buildIntersectionTriggerExplanationV2(ci, proofPayload)
            val triggerBridge = buildIntersectionTriggerBridgeV2(ci)

            val sourceConfinementProof =
                triggerExplanation.optJSONObject("source_confinement_proof")
                    ?.let { JSONObject(it.toString()) }
                    ?: JSONObject()

            val confrontationSummary = JSONObject().apply {
                put("kind", "INTERSECTION_CONFRONTATION_SUMMARY")
                put("interaction_kind", ci.interactionKind)
                put("final_resolution_kind", ci.finalResolutionKind)
                put("digit", ci.digit ?: JSONObject.NULL)
                put("source_house", JSONObject(ci.sourceHouse.toString()))
                put("cross_house", JSONObject(ci.targetHouse.toString()))
                put("target_house", JSONObject(ci.targetHouse.toString())) // backward-compatible alias
                put("overlap_cells", triggerPattern.optJSONArray("overlap_cells") ?: JSONArray())
                put("forbidden_cross_cells", triggerPattern.optJSONArray("forbidden_cross_cells") ?: JSONArray())
                put("ordinary_narrowing_count", 0)
                put(
                    "hero_elimination_count",
                    triggerPattern.optJSONArray("forbidden_cross_cells")?.length() ?: 0
                )
                put("source_confinement_proof", sourceConfinementProof)
                put(
                    "why_this_matters",
                    triggerBridge.optString("why_this_matters")
                        .takeIf { it.isNotBlank() }
                        ?: "This overlap changes what remains possible around the target."
                )
            }

            return JSONObject().apply {
                put("kind", "INTERSECTION_TRIGGER_PACKET")
                put("trigger_pattern", triggerPattern)
                put("trigger_explanation", triggerExplanation)
                put("trigger_bridge", triggerBridge)
                put("confrontation_summary", confrontationSummary)
            }
        }

        private fun buildSubsetWitnessPatternV2(
            stepObj: JSONObject,
            grid81: String?,
            cs: CanonicalSubset
        ): JSONObject {
            val subset = cs.subset
            val subtype = subset.subsetSubtype.lowercase()
            val kind = when {
                subset.subsetMode == "naked" -> "SUBSET_NAKED"
                "boxed" in subtype -> "SUBSET_HIDDEN_BOXED"
                else -> "SUBSET_HIDDEN"
            }

            val focusCellJson = cs.focusCell?.toJson()
            val lockedDigits = subset.lockedDigits.distinct().sorted()
            val sweepCells = subset.sweepCells.distinct().sorted()

            val cleanupDigits =
                cs.claimedCandidateDigits
                    .filter { it in lockedDigits }
                    .distinct()
                    .sorted()

            val targetRelation = JSONObject().apply {
                put(
                    "kind",
                    if (cs.eliminationKind == "CELL_CANDIDATE_DIGITS") {
                        "cleanup_target_cell"
                    } else {
                        "cleanup_target_house"
                    }
                )

                put("resolution_kind", cs.eliminationKind)

                if (focusCellJson != null) {
                    put("focus_cell", focusCellJson)
                }

                if (cs.targetDigit != null && cs.targetDigit in 1..9) {
                    put("target_digit", cs.targetDigit)
                }

                put("cleanup_digits", JSONArray().apply {
                    cleanupDigits.forEach { put(it) }
                })

                put("remaining_candidate_digits", JSONArray().apply {
                    cs.remainingCandidateDigits.forEach { put(it) }
                })

                put("claimed_candidate_digits", JSONArray().apply {
                    cs.claimedCandidateDigits.forEach { put(it) }
                })

                put("claimed_candidate_cells", JSONArray().apply {
                    cs.claimedCandidateCells.forEach { put(it.toJson()) }
                })

                put("remaining_candidate_cells", JSONArray().apply {
                    cs.remainingCandidateCells.forEach { put(it.toJson()) }
                })

                put(
                    "why_this_matters",
                    if (cs.eliminationKind == "CELL_CANDIDATE_DIGITS") {
                        "This subset can strip its locked digits away from the target cell, making that cell easier to resolve."
                    } else {
                        "This subset can narrow the allowed cells for the target digit in the affected house."
                    }
                )
            }

            val sweepRelation = JSONObject().apply {
                put("house", JSONObject(subset.house.toString()))
                put("locked_digits", JSONArray().apply {
                    lockedDigits.forEach { put(it) }
                })
                put("sweep_cells", JSONArray().apply {
                    sweepCells.forEach { idx ->
                        val r = (idx / 9) + 1
                        val c = (idx % 9) + 1
                        put(cellRef(r, c, idx))
                    }
                })
            }

            return JSONObject().apply {
                put("kind", kind)
                put("subset_mode", subset.subsetMode)
                put("subset_subtype", subset.subsetSubtype)
                put("house", JSONObject(subset.house.toString()))

                put("subset_cells", JSONArray().apply {
                    subset.subsetCells.distinct().sorted().forEach { idx ->
                        val r = (idx / 9) + 1
                        val c = (idx % 9) + 1
                        put(cellRef(r, c, idx))
                    }
                })

                put("locked_digits", JSONArray().apply {
                    lockedDigits.forEach { put(it) }
                })

                put("sweep_cells", JSONArray().apply {
                    sweepCells.forEach { idx ->
                        val r = (idx / 9) + 1
                        val c = (idx % 9) + 1
                        put(cellRef(r, c, idx))
                    }
                })

                put("target_relation", targetRelation)
                put("sweep_relation", sweepRelation)

                // Exact pre-step candidate content for each subset member cell.
                put("subset_cell_candidates", JSONArray().apply {
                    subset.subsetCells.distinct().sorted().forEach { idx ->
                        val r = (idx / 9) + 1
                        val c = (idx % 9) + 1
                        val digits = candidateDigitsBeforeForCell(stepObj, grid81, idx)
                            .filter { it in 1..9 }
                            .distinct()
                            .sorted()

                        put(JSONObject().apply {
                            put("cell", cellRef(r, c, idx))
                            put("digits", JSONArray().apply {
                                digits.forEach { put(it) }
                            })
                        })
                    }
                })
            }
        }

        private fun buildSubsetTriggerPatternV2(
            stepObj: JSONObject,
            grid81: String?,
            cs: CanonicalSubset
        ): JSONObject = JSONObject(buildSubsetWitnessPatternV2(stepObj, grid81, cs).toString())

        private fun gridDigitAtV2(grid81: String?, cellIndex: Int): Int? {
            if (grid81.isNullOrBlank()) return null
            if (cellIndex !in 0 until grid81.length) return null
            val ch = grid81[cellIndex]
            val digit = ch.digitToIntOrNull() ?: return null
            return digit.takeIf { it in 1..9 }
        }

        private fun rowIndex1to9ForCell(cellIndex: Int): Int = (cellIndex / 9) + 1

        private fun colIndex1to9ForCell(cellIndex: Int): Int = (cellIndex % 9) + 1

        private fun boxIndex1to9ForCell(cellIndex: Int): Int {
            val r = rowIndex1to9ForCell(cellIndex)
            val c = colIndex1to9ForCell(cellIndex)
            return ((r - 1) / 3) * 3 + ((c - 1) / 3) + 1
        }

        private fun simpleHouseRefV2(type: String, index1to9: Int): JSONObject =
            JSONObject().apply {
                put("type", type)
                put("index1to9", index1to9)
            }

        private fun findPeerWitnessForDigitV2(
            grid81: String?,
            cellIndex: Int,
            digit: Int
        ): JSONObject? {
            val row = rowIndex1to9ForCell(cellIndex)
            val col = colIndex1to9ForCell(cellIndex)
            val box = boxIndex1to9ForCell(cellIndex)

            // Row witness
            val rowStart = (row - 1) * 9
            for (c in 1..9) {
                val idx = rowStart + (c - 1)
                if (idx == cellIndex) continue
                if (gridDigitAtV2(grid81, idx) == digit) {
                    return JSONObject().apply {
                        put("digit", digit)
                        put("witness_cell", cellRef(row, c, idx))
                        put("via_house", simpleHouseRefV2("row", row))
                    }
                }
            }

            // Column witness
            for (r in 1..9) {
                val idx = (r - 1) * 9 + (col - 1)
                if (idx == cellIndex) continue
                if (gridDigitAtV2(grid81, idx) == digit) {
                    return JSONObject().apply {
                        put("digit", digit)
                        put("witness_cell", cellRef(r, col, idx))
                        put("via_house", simpleHouseRefV2("col", col))
                    }
                }
            }

            // Box witness
            val boxRowStart = ((box - 1) / 3) * 3 + 1
            val boxColStart = ((box - 1) % 3) * 3 + 1
            for (r in boxRowStart until boxRowStart + 3) {
                for (c in boxColStart until boxColStart + 3) {
                    val idx = (r - 1) * 9 + (c - 1)
                    if (idx == cellIndex) continue
                    if (gridDigitAtV2(grid81, idx) == digit) {
                        return JSONObject().apply {
                            put("digit", digit)
                            put("witness_cell", cellRef(r, c, idx))
                            put("via_house", simpleHouseRefV2("box", box))
                        }
                    }
                }
            }

            return null
        }

        private fun buildSubsetMemberCellCandidateProofV2(
            stepObj: JSONObject,
            grid81: String?,
            cellIndex: Int,
            lockedDigits: List<Int>
        ): JSONObject {
            val r = rowIndex1to9ForCell(cellIndex)
            val c = colIndex1to9ForCell(cellIndex)

            val remainingDigits =
                candidateDigitsBeforeForCell(stepObj, grid81, cellIndex)
                    .filter { it in 1..9 }
                    .distinct()
                    .sorted()

            val claimedDigits =
                (1..9)
                    .filterNot { it in remainingDigits }
                    .sorted()

            val lockedDigitsPresent =
                remainingDigits
                    .filter { it in lockedDigits }
                    .distinct()
                    .sorted()

            val witnessByDigit = JSONArray()
            val unsupportedDigits = JSONArray()

            claimedDigits.forEach { digit ->
                val witness = findPeerWitnessForDigitV2(grid81, cellIndex, digit)
                if (witness != null) {
                    witnessByDigit.put(witness)
                } else {
                    unsupportedDigits.put(digit)
                }
            }

            val status =
                if (unsupportedDigits.length() == 0) "ok"
                else "partial_peer_witness_coverage"

            return JSONObject().apply {
                put("cell", cellRef(r, c, cellIndex))
                put("explanation_kind", "CELL_CANDIDATE_DIGITS")

                put("remaining_candidate_digits", JSONArray().apply {
                    remainingDigits.forEach { put(it) }
                })

                put("claimed_candidate_digits", JSONArray().apply {
                    claimedDigits.forEach { put(it) }
                })

                put("locked_digits_present", JSONArray().apply {
                    lockedDigitsPresent.forEach { put(it) }
                })

                put("witness_by_digit", witnessByDigit)
                put("unsupported_claimed_digits", unsupportedDigits)
                put("status", status)

                put(
                    "proof_summary",
                    if (lockedDigitsPresent.isNotEmpty()) {
                        "All other digits are blocked around this cell, leaving ${lockedDigitsPresent.joinToString(" and ")} among its surviving candidates."
                    } else {
                        "This cell's peer evidence removes many other digits, leaving its current candidate set in place."
                    }
                )
            }
        }

        private fun buildSubsetTriggerExplanationV2(
            stepObj: JSONObject,
            grid81: String?,
            cs: CanonicalSubset,
            proofPayload: JSONObject
        ): JSONObject {
            val subset = cs.subset
            val lockedDigits = subset.lockedDigits.distinct().sorted()

            val memberProofs = JSONArray().apply {
                subset.subsetCells
                    .distinct()
                    .sorted()
                    .forEach { idx ->
                        put(
                            buildSubsetMemberCellCandidateProofV2(
                                stepObj = stepObj,
                                grid81 = grid81,
                                cellIndex = idx,
                                lockedDigits = lockedDigits
                            )
                        )
                    }
            }

            val support = proofPayload.optJSONObject("support") ?: JSONObject()

            return JSONObject().apply {
                put("kind", "SUBSET_TRIGGER_EXPLANATION")
                put("subset_mode", subset.subsetMode)
                put("subset_subtype", subset.subsetSubtype)
                put("house", JSONObject(subset.house.toString()))
                put("semantic_style", "pattern_member_cell_candidate_proofs")

                put("locked_digits", JSONArray().apply {
                    lockedDigits.forEach { put(it) }
                })

                put("pattern_member_proofs", memberProofs)

                put(
                    "downstream_support_snapshot",
                    JSONObject().apply {
                        val witnessByDigit = support.optJSONArray("witness_by_digit") ?: JSONArray()
                        put("witness_by_digit", JSONArray(witnessByDigit.toString()))
                    }
                )

                put("status", "ok")
            }
        }

        private fun buildSubsetTriggerBridgeV2(
            stepObj: JSONObject,
            grid81: String?,
            cs: CanonicalSubset
        ): JSONObject {
            val pattern = buildSubsetWitnessPatternV2(stepObj, grid81, cs)
            return JSONObject().apply {
                put("kind", "SUBSET_TRIGGER_BRIDGE")
                put("resolution_kind", cs.eliminationKind)
                put("target_relation", JSONObject((pattern.optJSONObject("target_relation") ?: JSONObject()).toString()))
                put("sweep_relation", JSONObject((pattern.optJSONObject("sweep_relation") ?: JSONObject()).toString()))
                put(
                    "why_this_matters",
                    pattern.optJSONObject("target_relation")?.optString("why_this_matters")
                        ?.takeIf { it.isNotBlank() }
                        ?: "This subset changes what remains possible around the target."
                )
            }
        }

        private fun buildSubsetTriggerPacketV2(
            stepObj: JSONObject,
            grid81: String?,
            cs: CanonicalSubset,
            proofPayload: JSONObject
        ): JSONObject {
            val triggerPattern = buildSubsetTriggerPatternV2(stepObj, grid81, cs)
            val triggerExplanation = buildSubsetTriggerExplanationV2(stepObj, grid81, cs, proofPayload)
            val triggerBridge = buildSubsetTriggerBridgeV2(stepObj, grid81, cs)

            val memberProofs = triggerExplanation.optJSONArray("pattern_member_proofs") ?: JSONArray()

            val confrontationSummary = JSONObject().apply {
                put("kind", "SUBSET_CONFRONTATION_SUMMARY")
                put("subset_mode", cs.subset.subsetMode)
                put("subset_subtype", cs.subset.subsetSubtype)
                put("resolution_kind", cs.eliminationKind)

                put("member_proof_count", memberProofs.length())

                put("pattern_member_proofs", JSONArray(memberProofs.toString()))

                put(
                    "target_relation",
                    JSONObject(
                        triggerBridge.optJSONObject("target_relation")?.toString() ?: "{}"
                    )
                )

                put(
                    "sweep_relation",
                    JSONObject(
                        triggerBridge.optJSONObject("sweep_relation")?.toString() ?: "{}"
                    )
                )

                put(
                    "why_this_matters",
                    triggerBridge.optString("why_this_matters")
                        .takeIf { it.isNotBlank() }
                        ?: "This subset changes what remains possible around the target."
                )
            }

            return JSONObject().apply {
                put("kind", "SUBSET_TRIGGER_PACKET")
                put("trigger_pattern", triggerPattern)
                put("trigger_explanation", triggerExplanation)
                put("trigger_bridge", triggerBridge)
                put("confrontation_summary", confrontationSummary)
            }
        }



        private fun buildSubsetProofPayloadV2(cs: CanonicalSubset): JSONObject {
            return if (cs.eliminationKind == "CELL_CANDIDATE_DIGITS") {
                val focusCell = cs.focusCell ?: return JSONObject()

                JSONObject().apply {
                    put("cell_outcome", JSONObject().apply {
                        put("cell", focusCell.toJson())
                        put("default_candidate_digits", JSONArray().apply {
                            cs.defaultCandidateDigits.forEach { put(it) }
                        })
                        put("claimed_candidate_digits", JSONArray().apply {
                            cs.claimedCandidateDigits.forEach { put(it) }
                        })
                        put("remaining_candidate_digits", JSONArray().apply {
                            cs.remainingCandidateDigits.forEach { put(it) }
                        })
                    })
                    put("support", JSONObject().apply {
                        put("witness_cells", JSONArray().apply {
                            cs.witnessCells.forEach { put(it.toJson()) }
                        })
                        put("witness_by_digit", JSONArray().apply {
                            cs.digitWitnesses.forEach { row ->
                                put(JSONObject().apply {
                                    put("digit", row.digit)
                                    put("witness", JSONObject().apply {
                                        put("kind", row.witnessKind)
                                        when (row.witnessKind) {
                                            "single_cell" -> {
                                                row.witnessCell?.let { put("cell", it.toJson()) }
                                                row.relation?.let { put("relation", it) }
                                            }
                                            "subset_group" -> {
                                                row.subsetKind?.let { put("subset_kind", it) }
                                                put("digits", JSONArray().apply {
                                                    row.subsetDigits.forEach { put(it) }
                                                })
                                                put("cells", JSONArray().apply {
                                                    row.subsetCells.forEach { put(it.toJson()) }
                                                })
                                                row.subsetHouse?.let { put("house", JSONObject(it.toString())) }
                                            }
                                        }
                                    })
                                })
                            }
                        })
                        put("eliminated_digits", JSONArray().apply {
                            cs.claimedCandidateDigits.forEach { put(it) }
                        })
                    })
                }
            } else {
                JSONObject().apply {
                    put("house_claim", JSONObject().apply {
                        put("digit", cs.targetDigit ?: JSONObject.NULL)
                        put("house", JSONObject(cs.primaryHouse.toString()))
                        put("default_candidate_cells", JSONArray().apply {
                            cs.defaultCandidateCells.forEach { put(it.toJson()) }
                        })
                        put("claimed_candidate_cells", JSONArray().apply {
                            cs.claimedCandidateCells.forEach { put(it.toJson()) }
                        })
                        put("remaining_candidate_cells", JSONArray().apply {
                            cs.remainingCandidateCells.forEach { put(it.toJson()) }
                        })
                    })
                    put("support", JSONObject().apply {
                        put("peer_cells", JSONArray().apply {
                            cs.claimedCandidateCells.forEach { put(it.toJson()) }
                        })
                        put("witness_by_cell", JSONArray().apply {
                            cs.cellWitnesses.forEach { row ->
                                put(JSONObject().apply {
                                    put("claimed_cell", row.claimedCell.toJson())
                                    put("witness", JSONObject().apply {
                                        put("kind", row.witnessKind)
                                        when (row.witnessKind) {
                                            "single_cell" -> {
                                                row.witnessCell?.let { put("cell", it.toJson()) }
                                                row.relation?.let { put("relation", it) }
                                            }
                                            "subset_group" -> {
                                                row.subsetKind?.let { put("subset_kind", it) }
                                                put("digits", JSONArray().apply {
                                                    row.subsetDigits.forEach { put(it) }
                                                })
                                                put("cells", JSONArray().apply {
                                                    row.subsetCells.forEach { put(it.toJson()) }
                                                })
                                                row.subsetHouse?.let { put("house", JSONObject(it.toString())) }
                                            }
                                        }
                                    })
                                })
                            }
                        })
                    })
                }
            }
        }



        private fun candidateDigitsBeforeForCell(stepObj: JSONObject, grid81: String?, cellIndex: Int): List<Int> {
            val snap = stepObj.optJSONObject("proof")
                ?.optJSONObject("candidates")
                ?.optJSONObject("snapshot_before")

            val mask = snap?.optInt(cellIndex.toString(), Int.MIN_VALUE) ?: Int.MIN_VALUE
            if (mask != Int.MIN_VALUE) return maskToDigits(mask)

            return computeCandidatesFromGrid(grid81, cellIndex)
        }



        private fun houseCellIndices(house: JSONObject): List<Int> {
            val type = house.optString("type")
            val idx = house.optInt("index1to9", -1)

            return when (type) {
                "row" -> if (idx in 1..9) ((idx - 1) * 9 until (idx - 1) * 9 + 9).toList() else emptyList()
                "col" -> if (idx in 1..9) (0 until 9).map { r -> r * 9 + (idx - 1) } else emptyList()
                "box" -> if (idx in 1..9) {
                    val br = ((idx - 1) / 3) * 3
                    val bc = ((idx - 1) % 3) * 3
                    buildList {
                        for (dr in 0..2) {
                            for (dc in 0..2) {
                                add((br + dr) * 9 + (bc + dc))
                            }
                        }
                    }
                } else emptyList()
                else -> emptyList()
            }
        }






        private fun parseCellIndexList(arr: JSONArray?): List<Int> {
            if (arr == null) return emptyList()
            val out = mutableListOf<Int>()
            for (i in 0 until arr.length()) {
                when (val raw = arr.opt(i)) {
                    is JSONObject -> {
                        val idx = raw.optInt("cellIndex", raw.optInt("cell_index", -1))
                        if (idx in 0..80) out += idx
                    }
                    is Number -> {
                        val idx = raw.toInt()
                        if (idx in 0..80) out += idx
                    }
                }
            }
            return out.distinct().sorted()
        }







        private fun maskToDigits(mask: Int): List<Int> =
            (1..9).filter { d -> (mask and (1 shl (d - 1))) != 0 }

        private fun computeCandidatesFromGrid(grid81: String?, cellIndex: Int): List<Int> {
            val g = grid81 ?: return emptyList()
            if (g.length != 81 || cellIndex !in 0..80) return emptyList()
            val ch = g[cellIndex]
            if (ch in '1'..'9') return listOf(ch.digitToInt())

            val r = cellIndex / 9
            val c = cellIndex % 9
            val used = mutableSetOf<Int>()

            for (cc in 0 until 9) {
                val v = g[r * 9 + cc]
                if (v in '1'..'9') used += v.digitToInt()
            }
            for (rr in 0 until 9) {
                val v = g[rr * 9 + c]
                if (v in '1'..'9') used += v.digitToInt()
            }

            val br = (r / 3) * 3
            val bc = (c / 3) * 3
            for (rr in br until br + 3) {
                for (cc in bc until bc + 3) {
                    val v = g[rr * 9 + cc]
                    if (v in '1'..'9') used += v.digitToInt()
                }
            }

            return (1..9).filterNot { it in used }
        }



        private fun peerIndicesOf(cellIndex: Int): List<Int> {
            if (cellIndex !in 0..80) return emptyList()

            val r = cellIndex / 9
            val c = cellIndex % 9
            val out = linkedSetOf<Int>()

            for (cc in 0 until 9) out += r * 9 + cc
            for (rr in 0 until 9) out += rr * 9 + c

            val br = (r / 3) * 3
            val bc = (c / 3) * 3
            for (rr in br until br + 3) {
                for (cc in bc until bc + 3) {
                    out += rr * 9 + cc
                }
            }

            out.remove(cellIndex)
            return out.toList().sorted()
        }
    }

    // ============================================================
    // Proof Validator V1
    // ============================================================

    object ProofValidatorV1 {

        fun validateNarrativeAtomsV1(packet: JSONObject?): JSONObject {
            val problems = JSONArray()
            val obj = packet ?: JSONObject()

            val schema = obj.optString("schema_version", "")
            if (schema != "narrative_atoms_v1") {
                problems.put(problem("bad_schema", null, "Expected narrative_atoms_v1 but got '$schema'"))
                return verdict(false, 0, problems)
            }

            val atoms = obj.optJSONArray("atoms")
            if (atoms == null || atoms.length() <= 0) {
                problems.put(problem("atoms_missing", null, "atoms[] missing or empty"))
                return verdict(false, 0, problems)
            }

            val atomCount = atoms.length()

            //val atomCount = atoms.length()

            val finalResolution = obj.optJSONObject("final_resolution") ?: JSONObject()
            validateFinalResolutionContract(obj, atoms, finalResolution, problems)
            validateStageShapeContract(atoms, problems)
            validateSetupAtomPurity(atoms, problems)

            for (i in 0 until atomCount) {
                val a = atoms.optJSONObject(i)
                if (a == null) {
                    problems.put(problem("atom_not_object", i, "atoms[$i] is not a JSONObject"))
                    continue
                }

                if (a.optString("schema_version") != "narrative_atom_v1") {
                    problems.put(problem("atom_bad_schema", i, "Expected narrative_atom_v1"))
                }

                val idx = a.optInt("index", -1)
                if (idx != i) {
                    problems.put(problem("atom_index_mismatch", i, "atom.index=$idx but array_index=$i"))
                }

                val beat = a.optString("beat_kind", "")
                val archetype = a.optString("archetype", "")
                if (beat.isBlank()) {
                    problems.put(problem("beat_kind_missing", i, "beat_kind missing"))
                }

                val focus = a.optJSONObject("focus")
                val targetCell = focus?.optJSONObject("target_cell")
                val td = focus?.optInt("target_digit", -1) ?: -1
                if (focus == null) {
                    problems.put(problem("focus_missing", i, "focus missing"))
                } else {
                    if (targetCell == null) {
                        problems.put(problem("focus_target_cell_missing", i, "focus.target_cell missing"))
                    }
                    if (td !in -1..9) {
                        problems.put(problem("focus_target_digit_invalid", i, "focus.target_digit invalid"))
                    }
                }

                val claim = a.optJSONObject("claim")
                val claimCode = claim?.optString("code", "").orEmpty()
                if (claimCode.isBlank()) {
                    problems.put(problem("claim_missing", i, "claim.code missing"))
                }

                val effects = a.optJSONObject("effects")
                if (effects == null) {
                    problems.put(problem("effects_missing", i, "effects missing"))
                } else {
                    if (effects.optJSONArray("eliminations") == null) {
                        problems.put(problem("effects_eliminations_missing", i, "effects.eliminations missing"))
                    }
                    if (effects.optJSONArray("placements") == null) {
                        problems.put(problem("effects_placements_missing", i, "effects.placements missing"))
                    }
                }

                val overlay = a.optJSONObject("overlay")
                val frameId = overlay?.optString("frame_id", "").orEmpty()
                val expectedFrame = "ov:atom:$i"
                if (frameId != expectedFrame) {
                    problems.put(problem("overlay_frame_id_mismatch", i, "overlay.frame_id='$frameId' expected '$expectedFrame'"))
                }

                val isNonClaimBeat = beat == NarrativeBeatKindV1.SPOTLIGHT.wire || beat == NarrativeBeatKindV1.TEACHING_NOTE.wire
                val isCommitBeat = beat == NarrativeBeatKindV1.COMMIT.wire
                val witnesses = a.optJSONArray("witnesses")

                if (!isNonClaimBeat && !isCommitBeat) {
                    if (witnesses == null || witnesses.length() <= 0) {
                        problems.put(problem("witnesses_missing", i, "$beat requires witnesses[]"))
                    }
                }

                when {
                    isCommitBeat -> validateCommitAtom(i, a, problems)
                    beat == NarrativeBeatKindV1.WITNESS_ELIMINATION.wire ->
                        validateWitnessEliminationAtom(i, a, claimCode, archetype, problems)
                    beat == NarrativeBeatKindV1.LOCK_IN.wire ->
                        validateLockInAtom(i, a, claimCode, archetype, problems)
                    beat == NarrativeBeatKindV1.TEACHING_NOTE.wire && archetype == NarrativeArchetypeV1.INTERSECTIONS.wire ->
                        validateIntersectionTeachingAtom(i, a, claimCode, problems)
                }
            }

            return verdict(problems.length() == 0, atomCount, problems)
        }

        private fun validateCommitAtom(i: Int, atom: JSONObject, problems: JSONArray) {
            val placements = atom.optJSONObject("effects")?.optJSONArray("placements")
            if (placements == null || placements.length() <= 0) {
                problems.put(problem("commit_placements_missing", i, "COMMIT requires at least one placement"))
            }
        }

        private fun validateLockInAtom(
            i: Int,
            atom: JSONObject,
            claimCode: String,
            archetype: String,
            problems: JSONArray
        ) {
            val witnesses = atom.optJSONArray("witnesses")
            val because = witnesses?.optJSONObject(0)?.optJSONObject("because")

            when {
                archetype == NarrativeArchetypeV1.HIDDEN_SINGLES.wire -> {
                    val rem = because?.optJSONObject("remaining_cell")?.optInt("cellIndex", -1) ?: -1
                    if (rem !in 0..80) {
                        problems.put(problem("lock_in_remaining_cell_missing", i, "Hidden-single LOCK_IN requires remaining_cell"))
                    }
                }
                archetype == NarrativeArchetypeV1.FULL_HOUSE.wire -> {
                    val rem = because?.optJSONObject("remaining_cell")?.optInt("cellIndex", -1) ?: -1
                    val digit = because?.optInt("remaining_digit", -1) ?: -1
                    if (rem !in 0..80 || digit !in 1..9) {
                        problems.put(problem("full_house_lock_in_missing", i, "Full-House LOCK_IN requires remaining_cell + remaining_digit"))
                    }
                }
                archetype == NarrativeArchetypeV1.NAKED_SINGLES.wire -> {
                    val rem = because?.optJSONObject("remaining_cell")?.optInt("cellIndex", -1) ?: -1
                    val digit = because?.optInt("remaining_digit", -1) ?: -1
                    if (rem !in 0..80 || digit !in 1..9) {
                        problems.put(problem("lock_in_remaining_digit_missing", i, "Naked-single LOCK_IN requires remaining_cell + remaining_digit"))
                    }
                }
                archetype == NarrativeArchetypeV1.INTERSECTIONS.wire &&
                        claimCode == NarrativeClaimCodeV1.DIGIT_LOCKED_TO_LINE_IN_BOX.wire -> {
                    val lockedCells = because?.optJSONArray("locked_cells")
                    val sourceHouse = because?.optJSONObject("source_house")
                    val targetHouse = because?.optJSONObject("target_house")
                    if (lockedCells == null || lockedCells.length() <= 0 || sourceHouse == null || targetHouse == null) {
                        problems.put(problem("intersection_lock_missing", i, "Intersection LOCK_IN requires locked_cells + source_house + target_house"))
                    }
                }
                claimCode == NarrativeClaimCodeV1.SUBSET_LOCKS_DIGITS.wire -> {
                    val subsetCells = because?.optJSONArray("subset_cells")
                    val lockedDigits = because?.optJSONArray("locked_digits")
                    if (subsetCells == null || subsetCells.length() <= 0 || lockedDigits == null || lockedDigits.length() <= 0) {
                        problems.put(problem("subset_lock_in_missing", i, "Subset LOCK_IN requires subset_cells + locked_digits"))
                    }
                }
            }
        }


        private fun validateIntersectionTeachingAtom(
            i: Int,
            atom: JSONObject,
            claimCode: String,
            problems: JSONArray
        ) {
            if (claimCode != "INTERSECTION_RESOLUTION" && claimCode != "TEACHING_NOTE_INTERSECTION") return

            val witnesses = atom.optJSONArray("witnesses")
            if (claimCode == "INTERSECTION_RESOLUTION") {
                val because = witnesses?.optJSONObject(0)?.optJSONObject("because")
                val sweepCells = because?.optJSONArray("sweep_cells")
                if (sweepCells == null || sweepCells.length() <= 0) {
                    problems.put(problem("intersection_resolution_missing", i, "Intersection resolution teaching atom requires sweep_cells"))
                }
            }
        }


        private fun validateWitnessEliminationAtom(
            i: Int,
            atom: JSONObject,
            claimCode: String,
            archetype: String,
            problems: JSONArray
        ) {
            val witnesses = atom.optJSONArray("witnesses")
            val because = witnesses?.optJSONObject(0)?.optJSONObject("because")
            val effects = atom.optJSONObject("effects")
            val elim = effects?.optJSONArray("eliminations")

            when {
                archetype == NarrativeArchetypeV1.HIDDEN_SINGLES.wire ||
                        archetype == NarrativeArchetypeV1.NAKED_SINGLES.wire -> {
                    val witnessIdx = because?.optJSONObject("witness_cell")?.optInt("cellIndex", -1) ?: -1
                    val peerIdx = because?.optJSONObject("explains_peer")?.optInt("cellIndex", -1) ?: -1
                    val relation = because?.optString("relation", "").orEmpty()
                    if (witnessIdx !in 0..80 || peerIdx !in 0..80 || relation.isBlank()) {
                        problems.put(problem("single_witness_shape_missing", i, "Singles WITNESS_ELIMINATION requires witness_cell + explains_peer + relation"))
                    }
                }

                claimCode == "INTERSECTION_SWEEP" -> {
                    val lockedCells = because?.optJSONArray("locked_cells")
                    val sweepCells = because?.optJSONArray("sweep_cells")
                    if (lockedCells == null || lockedCells.length() <= 0 || sweepCells == null || sweepCells.length() <= 0) {
                        problems.put(problem("intersection_witness_shape_missing", i, "Intersection sweep requires locked_cells + sweep_cells"))
                    }
                    if (elim == null || elim.length() <= 0) {
                        problems.put(problem("intersection_eliminations_missing", i, "Intersection sweep requires eliminations"))
                    }
                }

                claimCode == "SUBSET_SWEEP" -> {
                    val subsetCells = because?.optJSONArray("subset_cells")
                    val lockedDigits = because?.optJSONArray("locked_digits")
                    val sweepCells = because?.optJSONArray("sweep_cells")
                    if (subsetCells == null || lockedDigits == null || sweepCells == null ||
                        subsetCells.length() <= 0 || lockedDigits.length() <= 0 || sweepCells.length() <= 0
                    ) {
                        problems.put(problem("subset_sweep_shape_missing", i, "Naked subset sweep requires subset_cells + locked_digits + sweep_cells"))
                    }
                }

                claimCode == "HIDDEN_SUBSET_RESTRICT" -> {
                    val supportCells = because?.optJSONArray("support_cells")
                    val hiddenDigits = because?.optJSONArray("hidden_digits")
                    val removedDigits = because?.optJSONArray("removed_digits")
                    if (supportCells == null || hiddenDigits == null || removedDigits == null ||
                        supportCells.length() <= 0 || hiddenDigits.length() <= 0
                    ) {
                        problems.put(problem("hidden_subset_shape_missing", i, "Hidden subset restrict requires support_cells + hidden_digits + removed_digits"))
                    }
                }

                claimCode == "FISH_SWEEP" || archetype == NarrativeArchetypeV1.FISH.wire -> {
                    val corners = because?.optJSONArray("fish_corners") ?: because?.optJSONArray("corners")
                    val sweepCells = because?.optJSONArray("sweep_cells")
                    if ((corners == null || corners.length() <= 0) || (sweepCells == null || sweepCells.length() <= 0)) {
                        problems.put(problem("fish_shape_missing", i, "Fish sweep requires corners/fish structure + sweep_cells"))
                    }
                }

                claimCode == NarrativeClaimCodeV1.CELL_CANNOT_BE_DIGIT.wire &&
                        archetype == NarrativeArchetypeV1.WINGS.wire -> {
                    val hinge = because?.opt("hinge")
                    val pincers = because?.optJSONArray("pincers")
                    if (hinge == null || pincers == null || pincers.length() <= 0) {
                        problems.put(problem("wing_shape_missing", i, "Wing elimination requires hinge + pincers"))
                    }
                }

                claimCode == NarrativeClaimCodeV1.CONTRADICTION_IMPLES_NOT.wire ||
                        archetype == NarrativeArchetypeV1.CHAINS.wire -> {
                    val colorA = because?.optJSONArray("colorA")
                    val colorB = because?.optJSONArray("colorB")
                    if (colorA == null || colorB == null || colorA.length() <= 0 || colorB.length() <= 0) {
                        problems.put(problem("chain_shape_missing", i, "Chain contradiction requires colorA + colorB"))
                    }
                }

                else -> {
                    if (elim == null) {
                        problems.put(problem("generic_witness_missing_effects", i, "Witness elimination requires effects.eliminations"))
                    }
                }
            }
        }


        private fun isMeaningfulJsonObjectV2(obj: JSONObject?): Boolean =
            obj != null && obj.length() > 0

        private fun hasNonEmptyArrayFieldV2(obj: JSONObject?, key: String): Boolean =
            (obj?.optJSONArray(key)?.length() ?: 0) > 0

        private fun isValidCellRefV2(obj: JSONObject?): Boolean {
            if (obj == null) return false

            val cellIndex = obj.optInt("cellIndex", -1)
            val r = obj.optInt("r", -1)
            val c = obj.optInt("c", -1)

            return cellIndex in 0..80 &&
                    r in 1..9 &&
                    c in 1..9
        }

        private fun looksLikePlaceholderStatusV2(obj: JSONObject?): Boolean {
            val status = obj?.optString("status", "")?.trim().orEmpty()
            if (status.isBlank()) return false
            return status.startsWith("missing_") ||
                    status.startsWith("pending_") ||
                    status.contains("placeholder", ignoreCase = true)
        }

        private fun validateAdvancedAtom0SubsetPayloadV2(
            atom0Args: JSONObject,
            problems: JSONArray
        ) {
            val triggerPattern = atom0Args.optJSONObject("trigger_pattern")
            val triggerExplanation = atom0Args.optJSONObject("trigger_explanation")
            val triggerBridge = atom0Args.optJSONObject("trigger_bridge")
            val triggerPacket = atom0Args.optJSONObject("trigger_packet")
            val confrontationSummary = atom0Args.optJSONObject("confrontation_summary")

            if (triggerPattern == null) {
                problems.put(
                    problem(
                        "atom0_subset_trigger_pattern_missing",
                        0,
                        "SUBSETS atom0 must include trigger_pattern"
                    )
                )
            } else {
                if (triggerPattern.optString("subset_mode").isBlank()) {
                    problems.put(
                        problem(
                            "atom0_subset_mode_missing",
                            0,
                            "SUBSETS atom0 trigger_pattern must include subset_mode"
                        )
                    )
                }
                if (triggerPattern.optString("subset_subtype").isBlank()) {
                    problems.put(
                        problem(
                            "atom0_subset_subtype_missing",
                            0,
                            "SUBSETS atom0 trigger_pattern must include subset_subtype"
                        )
                    )
                }
                if (!isValidHouse(triggerPattern.optJSONObject("house"))) {
                    problems.put(
                        problem(
                            "atom0_subset_house_missing",
                            0,
                            "SUBSETS atom0 trigger_pattern must include a valid house"
                        )
                    )
                }
                if (!hasNonEmptyArrayFieldV2(triggerPattern, "subset_cells")) {
                    problems.put(
                        problem(
                            "atom0_subset_cells_missing",
                            0,
                            "SUBSETS atom0 trigger_pattern must include non-empty subset_cells"
                        )
                    )
                }
                if (!hasNonEmptyArrayFieldV2(triggerPattern, "locked_digits")) {
                    problems.put(
                        problem(
                            "atom0_subset_locked_digits_missing",
                            0,
                            "SUBSETS atom0 trigger_pattern must include non-empty locked_digits"
                        )
                    )
                }
            }

            if (triggerExplanation == null) {
                problems.put(
                    problem(
                        "atom0_subset_trigger_explanation_missing",
                        0,
                        "SUBSETS atom0 must include trigger_explanation"
                    )
                )
            } else {
                if (looksLikePlaceholderStatusV2(triggerExplanation)) {
                    problems.put(
                        problem(
                            "atom0_subset_trigger_explanation_placeholder",
                            0,
                            "SUBSETS atom0 trigger_explanation must be populated, not placeholder-only"
                        )
                    )
                }

                if (!hasNonEmptyArrayFieldV2(triggerExplanation, "pattern_member_proofs")) {
                    problems.put(
                        problem(
                            "atom0_subset_pattern_member_proofs_missing",
                            0,
                            "SUBSETS atom0 trigger_explanation must include non-empty pattern_member_proofs"
                        )
                    )
                } else {
                    val proofs = triggerExplanation.optJSONArray("pattern_member_proofs") ?: JSONArray()
                    for (i in 0 until proofs.length()) {
                        val proof = proofs.optJSONObject(i) ?: continue

                        if (!isValidCellRefV2(proof.optJSONObject("cell"))) {
                            problems.put(
                                problem(
                                    "atom0_subset_pattern_member_cell_missing",
                                    0,
                                    "Each subset pattern_member_proof must include a valid cell"
                                )
                            )
                        }

                        if (proof.optString("explanation_kind") != "CELL_CANDIDATE_DIGITS") {
                            problems.put(
                                problem(
                                    "atom0_subset_pattern_member_wrong_explanation_kind",
                                    0,
                                    "Each subset pattern_member_proof must use CELL_CANDIDATE_DIGITS explanation_kind"
                                )
                            )
                        }

                        if (!hasNonEmptyArrayFieldV2(proof, "remaining_candidate_digits")) {
                            problems.put(
                                problem(
                                    "atom0_subset_pattern_member_remaining_digits_missing",
                                    0,
                                    "Each subset pattern_member_proof must include remaining_candidate_digits"
                                )
                            )
                        }
                    }
                }
            }

            if (triggerBridge == null) {
                problems.put(
                    problem(
                        "atom0_subset_trigger_bridge_missing",
                        0,
                        "SUBSETS atom0 must include trigger_bridge"
                    )
                )
            } else {
                if (looksLikePlaceholderStatusV2(triggerBridge)) {
                    problems.put(
                        problem(
                            "atom0_subset_trigger_bridge_placeholder",
                            0,
                            "SUBSETS atom0 trigger_bridge must be populated, not placeholder-only"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerBridge.optJSONObject("target_relation"))) {
                    problems.put(
                        problem(
                            "atom0_subset_target_relation_missing",
                            0,
                            "SUBSETS atom0 trigger_bridge must include target_relation"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerBridge.optJSONObject("sweep_relation"))) {
                    problems.put(
                        problem(
                            "atom0_subset_sweep_relation_missing",
                            0,
                            "SUBSETS atom0 trigger_bridge must include sweep_relation"
                        )
                    )
                }
            }

            if (triggerPacket == null) {
                problems.put(
                    problem(
                        "atom0_subset_trigger_packet_missing",
                        0,
                        "SUBSETS atom0 must include trigger_packet"
                    )
                )
            } else {
                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("trigger_pattern"))) {
                    problems.put(
                        problem(
                            "atom0_subset_trigger_packet_pattern_missing",
                            0,
                            "SUBSETS trigger_packet must include trigger_pattern"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("trigger_explanation"))) {
                    problems.put(
                        problem(
                            "atom0_subset_trigger_packet_explanation_missing",
                            0,
                            "SUBSETS trigger_packet must include trigger_explanation"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("trigger_bridge"))) {
                    problems.put(
                        problem(
                            "atom0_subset_trigger_packet_bridge_missing",
                            0,
                            "SUBSETS trigger_packet must include trigger_bridge"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("confrontation_summary"))) {
                    problems.put(
                        problem(
                            "atom0_subset_trigger_packet_confrontation_summary_missing",
                            0,
                            "SUBSETS trigger_packet must include confrontation_summary"
                        )
                    )
                }
            }

            if (confrontationSummary == null) {
                problems.put(
                    problem(
                        "atom0_subset_confrontation_summary_missing",
                        0,
                        "SUBSETS atom0 must include confrontation_summary"
                    )
                )
            } else {
                if (looksLikePlaceholderStatusV2(confrontationSummary)) {
                    problems.put(
                        problem(
                            "atom0_subset_confrontation_summary_placeholder",
                            0,
                            "SUBSETS confrontation_summary must be populated, not placeholder-only"
                        )
                    )
                }

                if ((confrontationSummary.optInt("member_proof_count", 0) <= 0) &&
                    !hasNonEmptyArrayFieldV2(confrontationSummary, "pattern_member_proofs")
                ) {
                    problems.put(
                        problem(
                            "atom0_subset_confrontation_summary_member_proofs_missing",
                            0,
                            "SUBSETS confrontation_summary must expose member proof coverage"
                        )
                    )
                }
            }
        }

        private fun validateAdvancedAtom0IntersectionPayloadV2(
            atom0Args: JSONObject,
            problems: JSONArray
        ) {
            val triggerPattern = atom0Args.optJSONObject("trigger_pattern")
            val triggerExplanation = atom0Args.optJSONObject("trigger_explanation")
            val triggerBridge = atom0Args.optJSONObject("trigger_bridge")
            val triggerPacket = atom0Args.optJSONObject("trigger_packet")
            val confrontationSummary = atom0Args.optJSONObject("confrontation_summary")

            if (triggerPattern == null) {
                problems.put(
                    problem(
                        "atom0_intersection_trigger_pattern_missing",
                        0,
                        "INTERSECTIONS atom0 must include trigger_pattern"
                    )
                )
            } else {
                if (triggerPattern.optString("interaction_kind").isBlank()) {
                    problems.put(
                        problem(
                            "atom0_intersection_interaction_kind_missing",
                            0,
                            "INTERSECTIONS atom0 trigger_pattern must include interaction_kind"
                        )
                    )
                }

                if (!isValidHouse(triggerPattern.optJSONObject("source_house"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_source_house_missing",
                            0,
                            "INTERSECTIONS atom0 trigger_pattern must include valid source_house"
                        )
                    )
                }

                if (!isValidHouse(triggerPattern.optJSONObject("target_house"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_target_house_missing",
                            0,
                            "INTERSECTIONS atom0 trigger_pattern must include valid target_house"
                        )
                    )
                }
            }

            if (triggerExplanation == null) {
                problems.put(
                    problem(
                        "atom0_intersection_trigger_explanation_missing",
                        0,
                        "INTERSECTIONS atom0 must include trigger_explanation"
                    )
                )
            } else {
                if (looksLikePlaceholderStatusV2(triggerExplanation)) {
                    problems.put(
                        problem(
                            "atom0_intersection_trigger_explanation_placeholder",
                            0,
                            "INTERSECTIONS atom0 trigger_explanation must be populated, not placeholder-only"
                        )
                    )
                }

                val sourceConfinementProof =
                    triggerExplanation.optJSONObject("source_confinement_proof")

                if (!isMeaningfulJsonObjectV2(sourceConfinementProof)) {
                    problems.put(
                        problem(
                            "atom0_intersection_source_confinement_proof_missing",
                            0,
                            "INTERSECTIONS atom0 trigger_explanation must include populated source_confinement_proof"
                        )
                    )
                } else if (looksLikePlaceholderStatusV2(sourceConfinementProof)) {
                    problems.put(
                        problem(
                            "atom0_intersection_source_confinement_proof_placeholder",
                            0,
                            "INTERSECTIONS atom0 source_confinement_proof must be meaningfully populated"
                        )
                    )
                }
            }

            if (triggerBridge == null) {
                problems.put(
                    problem(
                        "atom0_intersection_trigger_bridge_missing",
                        0,
                        "INTERSECTIONS atom0 must include trigger_bridge"
                    )
                )
            } else {
                if (looksLikePlaceholderStatusV2(triggerBridge)) {
                    problems.put(
                        problem(
                            "atom0_intersection_trigger_bridge_placeholder",
                            0,
                            "INTERSECTIONS atom0 trigger_bridge must be populated, not placeholder-only"
                        )
                    )
                }

                if (triggerBridge.optString("final_resolution_kind").isBlank() &&
                    triggerBridge.optString("downstream_resolution_kind").isBlank()
                ) {
                    problems.put(
                        problem(
                            "atom0_intersection_bridge_resolution_kind_missing",
                            0,
                            "INTERSECTIONS atom0 trigger_bridge must include downstream/final resolution kind"
                        )
                    )
                }
            }

            if (triggerPacket == null) {
                problems.put(
                    problem(
                        "atom0_intersection_trigger_packet_missing",
                        0,
                        "INTERSECTIONS atom0 must include trigger_packet"
                    )
                )
            } else {
                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("trigger_pattern"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_trigger_packet_pattern_missing",
                            0,
                            "INTERSECTIONS trigger_packet must include trigger_pattern"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("trigger_explanation"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_trigger_packet_explanation_missing",
                            0,
                            "INTERSECTIONS trigger_packet must include trigger_explanation"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("trigger_bridge"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_trigger_packet_bridge_missing",
                            0,
                            "INTERSECTIONS trigger_packet must include trigger_bridge"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(triggerPacket.optJSONObject("confrontation_summary"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_trigger_packet_confrontation_summary_missing",
                            0,
                            "INTERSECTIONS trigger_packet must include confrontation_summary"
                        )
                    )
                }
            }

            if (confrontationSummary == null) {
                problems.put(
                    problem(
                        "atom0_intersection_confrontation_summary_missing",
                        0,
                        "INTERSECTIONS atom0 must include confrontation_summary"
                    )
                )
            } else {
                if (looksLikePlaceholderStatusV2(confrontationSummary)) {
                    problems.put(
                        problem(
                            "atom0_intersection_confrontation_summary_placeholder",
                            0,
                            "INTERSECTIONS confrontation_summary must be populated, not placeholder-only"
                        )
                    )
                }

                if (!isMeaningfulJsonObjectV2(confrontationSummary.optJSONObject("source_confinement_proof"))) {
                    problems.put(
                        problem(
                            "atom0_intersection_confrontation_summary_source_confinement_missing",
                            0,
                            "INTERSECTIONS confrontation_summary must expose source_confinement_proof"
                        )
                    )
                }
            }
        }


        fun buildAtom0AuditSnapshotV2(packet: JSONObject): JSONObject {
            val evidence = packet.optJSONObject("evidence") ?: JSONObject()
            val truth = evidence.optJSONObject("narrative_truth_v2") ?: JSONObject()
            val atomsDoc = evidence.optJSONObject("narrative_atoms_v1") ?: JSONObject()
            val atoms = atomsDoc.optJSONArray("atoms") ?: JSONArray()
            val atom0 = atoms.optJSONObject(0) ?: JSONObject()

            val finalResolution = truth.optJSONObject("final_resolution") ?: JSONObject()
            val truthTriggerPattern = truth.optJSONObject("trigger_pattern") ?: JSONObject()
            val truthTriggerExplanation = truth.optJSONObject("trigger_explanation") ?: JSONObject()
            val truthTriggerBridge = truth.optJSONObject("trigger_bridge") ?: JSONObject()
            val truthTriggerPacket = truth.optJSONObject("trigger_packet") ?: JSONObject()

            val atom0Args =
                atom0.optJSONObject("claim")
                    ?.optJSONObject("args") ?: JSONObject()

            val atom0TriggerPattern = atom0Args.optJSONObject("trigger_pattern") ?: JSONObject()
            val atom0TriggerExplanation = atom0Args.optJSONObject("trigger_explanation") ?: JSONObject()
            val atom0TriggerBridge = atom0Args.optJSONObject("trigger_bridge") ?: JSONObject()
            val atom0TriggerPacket = atom0Args.optJSONObject("trigger_packet") ?: JSONObject()
            val atom0ConfrontationSummary = atom0Args.optJSONObject("confrontation_summary") ?: JSONObject()

            return JSONObject().apply {
                put("kind", "ATOM0_AUDIT_SNAPSHOT_V2")

                put("archetype", atomsDoc.optString("archetype"))
                put("atom0_index", atom0.optInt("index", -1))
                put("atom0_setup_role", atom0Args.optString("setup_role"))
                put("final_resolution_kind", finalResolution.optString("kind"))

                put(
                    "truth_has_trigger_pattern",
                    truthTriggerPattern.length() > 0
                )
                put(
                    "truth_has_trigger_explanation",
                    truthTriggerExplanation.length() > 0
                )
                put(
                    "truth_has_trigger_bridge",
                    truthTriggerBridge.length() > 0
                )
                put(
                    "truth_has_trigger_packet",
                    truthTriggerPacket.length() > 0
                )

                put(
                    "atom0_has_trigger_pattern",
                    atom0TriggerPattern.length() > 0
                )
                put(
                    "atom0_has_trigger_explanation",
                    atom0TriggerExplanation.length() > 0
                )
                put(
                    "atom0_has_trigger_bridge",
                    atom0TriggerBridge.length() > 0
                )
                put(
                    "atom0_has_trigger_packet",
                    atom0TriggerPacket.length() > 0
                )
                put(
                    "atom0_has_confrontation_summary",
                    atom0ConfrontationSummary.length() > 0
                )

                put(
                    "truth_trigger_kind",
                    truthTriggerPattern.optString("kind").takeIf { it.isNotBlank() } ?: JSONObject.NULL
                )
                put(
                    "atom0_trigger_kind",
                    atom0TriggerPattern.optString("kind").takeIf { it.isNotBlank() } ?: JSONObject.NULL
                )

                put(
                    "truth_vs_atom0_trigger_kind_match",
                    truthTriggerPattern.optString("kind") == atom0TriggerPattern.optString("kind") &&
                            truthTriggerPattern.optString("kind").isNotBlank()
                )

                put(
                    "truth_vs_atom0_resolution_kind_match",
                    finalResolution.optString("kind").isNotBlank() &&
                            finalResolution.optString("kind") ==
                            atom0Args.optJSONObject("atom0_invariant_contract")
                                ?.optJSONObject("target_alignment")
                                ?.optString("resolution_kind")
                )

                put(
                    "truth_trigger_packet_kind",
                    truthTriggerPacket.optString("kind").takeIf { it.isNotBlank() } ?: JSONObject.NULL
                )
                put(
                    "atom0_trigger_packet_kind",
                    atom0TriggerPacket.optString("kind").takeIf { it.isNotBlank() } ?: JSONObject.NULL
                )
            }
        }

        private fun validateTruthAtom0ConsistencyV2(packet: JSONObject): JSONArray {
            val problems = JSONArray()

            val evidence = packet.optJSONObject("evidence") ?: return problems
            val truth = evidence.optJSONObject("narrative_truth_v2") ?: return problems
            val atomsDoc = evidence.optJSONObject("narrative_atoms_v1") ?: return problems
            val atoms = atomsDoc.optJSONArray("atoms") ?: return problems
            val atom0 = atoms.optJSONObject(0) ?: return problems

            val truthFinalResolution = truth.optJSONObject("final_resolution") ?: JSONObject()
            val truthTriggerPattern = truth.optJSONObject("trigger_pattern") ?: JSONObject()
            val truthTriggerExplanation = truth.optJSONObject("trigger_explanation") ?: JSONObject()
            val truthTriggerBridge = truth.optJSONObject("trigger_bridge") ?: JSONObject()
            val truthTriggerPacket = truth.optJSONObject("trigger_packet") ?: JSONObject()

            val atom0Args =
                atom0.optJSONObject("claim")
                    ?.optJSONObject("args") ?: JSONObject()

            val atom0TriggerPattern = atom0Args.optJSONObject("trigger_pattern") ?: JSONObject()
            val atom0TriggerExplanation = atom0Args.optJSONObject("trigger_explanation") ?: JSONObject()
            val atom0TriggerBridge = atom0Args.optJSONObject("trigger_bridge") ?: JSONObject()
            val atom0TriggerPacket = atom0Args.optJSONObject("trigger_packet") ?: JSONObject()

            if (truthTriggerPattern.length() > 0 && atom0TriggerPattern.length() == 0) {
                problems.put(
                    problem(
                        "atom0_missing_truth_trigger_pattern",
                        0,
                        "truth carries trigger_pattern but atom0 does not expose it"
                    )
                )
            }

            if (truthTriggerExplanation.length() > 0 && atom0TriggerExplanation.length() == 0) {
                problems.put(
                    problem(
                        "atom0_missing_truth_trigger_explanation",
                        0,
                        "truth carries trigger_explanation but atom0 does not expose it"
                    )
                )
            }

            if (truthTriggerBridge.length() > 0 && atom0TriggerBridge.length() == 0) {
                problems.put(
                    problem(
                        "atom0_missing_truth_trigger_bridge",
                        0,
                        "truth carries trigger_bridge but atom0 does not expose it"
                    )
                )
            }

            if (truthTriggerPacket.length() > 0 && atom0TriggerPacket.length() == 0) {
                problems.put(
                    problem(
                        "atom0_missing_truth_trigger_packet",
                        0,
                        "truth carries trigger_packet but atom0 does not expose it"
                    )
                )
            }

            val truthTriggerKind = truthTriggerPattern.optString("kind")
            val atom0TriggerKind = atom0TriggerPattern.optString("kind")
            if (truthTriggerKind.isNotBlank() &&
                atom0TriggerKind.isNotBlank() &&
                truthTriggerKind != atom0TriggerKind
            ) {
                problems.put(
                    problem(
                        "truth_atom0_trigger_kind_mismatch",
                        0,
                        "truth trigger_pattern.kind must match atom0 trigger_pattern.kind"
                    )
                )
            }

            val truthResolutionKind = truthFinalResolution.optString("kind")
            val atom0ResolutionKind =
                atom0Args.optJSONObject("atom0_invariant_contract")
                    ?.optJSONObject("target_alignment")
                    ?.optString("resolution_kind").orEmpty()

            if (truthResolutionKind.isNotBlank() &&
                atom0ResolutionKind.isNotBlank() &&
                truthResolutionKind != atom0ResolutionKind
            ) {
                problems.put(
                    problem(
                        "truth_atom0_resolution_kind_mismatch",
                        0,
                        "truth final_resolution.kind must match atom0 invariant target_alignment.resolution_kind"
                    )
                )
            }

            return problems
        }


        fun validateFinalResolutionContract(
            packet: JSONObject,
            atoms: JSONArray,
            finalResolution: JSONObject,
            problems: JSONArray
        ) {
            if (finalResolution.length() <= 0) {
                problems.put(
                    problem(
                        "final_resolution_missing",
                        null,
                        "Packet must carry final_resolution"
                    )
                )

                if (atoms.length() > 0) {
                    problems.put(
                        problem(
                            "contract_coverage_missing_final_resolution_with_atoms",
                            null,
                            "Packet emitted atoms without final_resolution coverage"
                        )
                    )
                }
                return
            }

            val kind = finalResolution.optString("kind", "")
            val allowedKinds = setOf(
                "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
                "CELL_CANDIDATE_DIGITS"
            )

            if (kind !in allowedKinds) {
                problems.put(
                    problem(
                        "final_resolution_kind_invalid",
                        null,
                        "final_resolution.kind must be HOUSE_CANDIDATE_CELLS_FOR_DIGIT or CELL_CANDIDATE_DIGITS"
                    )
                )
            }

            val finalFocus = finalResolution.optJSONObject("focus_cell")
            val finalFocusIdx = finalFocus?.optInt("cellIndex", -1) ?: -1
            if (finalFocusIdx !in 0..80) {
                problems.put(
                    problem(
                        "final_resolution_focus_missing",
                        null,
                        "final_resolution.focus_cell missing or invalid"
                    )
                )
            }

            val finalDigit = finalResolution.optInt("digit", -1)
            if (finalDigit !in 1..9) {
                problems.put(
                    problem(
                        "final_resolution_digit_missing",
                        null,
                        "final_resolution.digit missing or invalid"
                    )
                )
            }

            val atom0 = atoms.optJSONObject(0)
            if (atom0 == null) {
                problems.put(problem("atom0_missing", null, "atoms[0] missing"))
                return
            }

            val atom0FocusIdx =
                atom0.optJSONObject("focus")
                    ?.optJSONObject("target_cell")
                    ?.optInt("cellIndex", -1)
                    ?: -1

            if (finalFocusIdx in 0..80 && atom0FocusIdx != finalFocusIdx) {
                problems.put(
                    problem(
                        "atom0_focus_mismatch_final_resolution",
                        0,
                        "atom0 focus cell must equal final_resolution.focus_cell"
                    )
                )
            }

            val atom0PrimaryHouse =
                atom0.optJSONObject("focus")
                    ?.optJSONObject("primary_house")

            val finalPrimaryHouse =
                finalResolution.optJSONObject("primary_house")

            val atom0Archetype = atom0.optString("archetype", "")
            val atom0InvariantContract =
                atom0.optJSONObject("claim")
                    ?.optJSONObject("args")
                    ?.optJSONObject("atom0_invariant_contract")

            if (atom0InvariantContract == null) {
                problems.put(
                    problem(
                        "atom0_invariant_contract_missing",
                        0,
                        "atom0 must carry claim.args.atom0_invariant_contract"
                    )
                )
            } else {
                val targetAlignment = atom0InvariantContract.optJSONObject("target_alignment")
                if (targetAlignment == null) {
                    problems.put(
                        problem(
                            "atom0_invariant_target_alignment_missing",
                            0,
                            "atom0_invariant_contract must carry target_alignment"
                        )
                    )
                } else {
                    if (!targetAlignment.optBoolean("requires_focus_cell_match", false)) {
                        problems.put(
                            problem(
                                "atom0_invariant_focus_alignment_flag_missing",
                                0,
                                "atom0 target_alignment must require focus-cell alignment"
                            )
                        )
                    }

                    if (!targetAlignment.optBoolean("requires_primary_house_match_when_house_based", false)) {
                        problems.put(
                            problem(
                                "atom0_invariant_primary_house_alignment_flag_missing",
                                0,
                                "atom0 target_alignment must require primary-house alignment for house-based resolutions"
                            )
                        )
                    }
                }

                val isAdvancedAtom0 =
                    atom0Archetype != NarrativeArchetypeV1.HIDDEN_SINGLES.wire &&
                            atom0Archetype != NarrativeArchetypeV1.NAKED_SINGLES.wire &&
                            atom0Archetype != NarrativeArchetypeV1.FULL_HOUSE.wire

                if (isAdvancedAtom0) {
                    if (!atom0InvariantContract.optBoolean("advanced_trigger_required", false)) {
                        problems.put(
                            problem(
                                "atom0_advanced_trigger_requirement_missing",
                                0,
                                "Advanced atom0 must declare advanced_trigger_required=true"
                            )
                        )
                    }

                    val atom0Args =
                        atom0.optJSONObject("claim")
                            ?.optJSONObject("args")

                    if (atom0Args?.optString("setup_role") != "advanced_trigger_setup") {
                        problems.put(
                            problem(
                                "atom0_advanced_setup_role_missing",
                                0,
                                "Advanced atom0 must expose setup_role=advanced_trigger_setup"
                            )
                        )
                    }

                    if (atom0Args?.optJSONObject("trigger_pattern") == null) {
                        problems.put(
                            problem(
                                "atom0_trigger_pattern_missing",
                                0,
                                "Advanced atom0 must expose claim.args.trigger_pattern"
                            )
                        )
                    }

                    if (atom0Args?.optJSONObject("trigger_explanation") == null) {
                        problems.put(
                            problem(
                                "atom0_trigger_explanation_missing",
                                0,
                                "Advanced atom0 must expose claim.args.trigger_explanation"
                            )
                        )
                    }

                    if (atom0Args?.optJSONObject("trigger_bridge") == null) {
                        problems.put(
                            problem(
                                "atom0_trigger_bridge_missing",
                                0,
                                "Advanced atom0 must expose claim.args.trigger_bridge"
                            )
                        )
                    }

                    val advancedSetupPayload = atom0Args?.optJSONObject("advanced_setup_payload")
                    if (advancedSetupPayload == null) {
                        problems.put(
                            problem(
                                "atom0_advanced_setup_payload_missing",
                                0,
                                "Advanced atom0 must expose claim.args.advanced_setup_payload"
                            )
                        )
                    } else {
                        if (advancedSetupPayload.optString("setup_role") != "advanced_trigger_setup") {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_role_mismatch",
                                    0,
                                    "advanced_setup_payload must declare setup_role=advanced_trigger_setup"
                                )
                            )
                        }

                        if (advancedSetupPayload.optJSONObject("trigger_pattern") == null) {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_trigger_pattern_missing",
                                    0,
                                    "advanced_setup_payload must include trigger_pattern"
                                )
                            )
                        }

                        if (advancedSetupPayload.optJSONObject("trigger_explanation") == null) {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_trigger_explanation_missing",
                                    0,
                                    "advanced_setup_payload must include trigger_explanation"
                                )
                            )
                        }

                        if (advancedSetupPayload.optJSONObject("trigger_bridge") == null) {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_trigger_bridge_missing",
                                    0,
                                    "advanced_setup_payload must include trigger_bridge"
                                )
                            )
                        }

                        if (advancedSetupPayload.optJSONObject("intro_narration_contract") == null) {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_intro_narration_contract_missing",
                                    0,
                                    "advanced_setup_payload must include intro_narration_contract"
                                )
                            )
                        }

                        val introOverlayContract =
                            advancedSetupPayload.optJSONObject("intro_overlay_contract")

                        if (introOverlayContract == null) {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_intro_overlay_contract_missing",
                                    0,
                                    "advanced_setup_payload must include intro_overlay_contract"
                                )
                            )
                        } else {
                            val atom0Overlay = atom0.optJSONObject("overlay")

                            if (atom0Overlay == null) {
                                problems.put(
                                    problem(
                                        "atom0_overlay_missing",
                                        0,
                                        "Advanced atom0 must expose overlay"
                                    )
                                )
                            } else {
                                val overlayIntent = atom0Overlay.optString("intent")
                                val expectedIntent =
                                    introOverlayContract.optString("intent")
                                        .ifBlank { OverlayIntentV1.SHOW_SPOTLIGHT.wire }

                                if (overlayIntent != expectedIntent) {
                                    problems.put(
                                        problem(
                                            "atom0_overlay_intent_mismatch",
                                            0,
                                            "atom0.overlay.intent='$overlayIntent' expected '$expectedIntent'"
                                        )
                                    )
                                }

                                val overlaySetupRole = atom0Overlay.optString("setup_role")
                                if (overlaySetupRole != "advanced_trigger_setup") {
                                    problems.put(
                                        problem(
                                            "atom0_overlay_setup_role_mismatch",
                                            0,
                                            "atom0.overlay.setup_role must be advanced_trigger_setup"
                                        )
                                    )
                                }

                                val overlaySetupVariant = atom0Overlay.optString("setup_variant")
                                val expectedSetupVariant =
                                    introOverlayContract.optString("setup_variant")

                                if (expectedSetupVariant.isNotBlank() &&
                                    overlaySetupVariant != expectedSetupVariant
                                ) {
                                    problems.put(
                                        problem(
                                            "atom0_overlay_setup_variant_mismatch",
                                            0,
                                            "atom0.overlay.setup_variant='$overlaySetupVariant' expected '$expectedSetupVariant'"
                                        )
                                    )
                                }

                                val contractFlags = listOf(
                                    "show_target_focus",
                                    "show_subset_house",
                                    "show_subset_cells",
                                    "show_subset_candidates",
                                    "show_source_house",
                                    "show_target_house",
                                    "show_pattern_cells",
                                    "show_sweep_cells",
                                    "show_blocker_network",
                                    "show_resolution_collapse",
                                    "show_commit"
                                )

                                contractFlags.forEach { flag ->
                                    if (introOverlayContract.has(flag)) {
                                        val expected = introOverlayContract.optBoolean(flag)
                                        val actual = atom0Overlay.optBoolean(flag, !expected)
                                        if (actual != expected) {
                                            problems.put(
                                                problem(
                                                    "atom0_overlay_flag_mismatch_$flag",
                                                    0,
                                                    "atom0.overlay.$flag=$actual expected $expected"
                                                )
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        val introDerived = advancedSetupPayload.optJSONObject("intro_derived")
                        if (introDerived == null) {
                            problems.put(
                                problem(
                                    "atom0_advanced_setup_payload_intro_derived_missing",
                                    0,
                                    "advanced_setup_payload must include intro_derived"
                                )
                            )
                        } else {
                            val introRoute = advancedSetupPayload.optJSONArray("intro_route")
                            if (!hasNonEmptyArrayFieldV2(advancedSetupPayload, "intro_route")) {
                                problems.put(
                                    problem(
                                        "atom0_advanced_setup_payload_intro_route_missing",
                                        0,
                                        "advanced_setup_payload must include non-empty intro_route"
                                    )
                                )
                            } else {
                                val introNarrationContract =
                                    advancedSetupPayload.optJSONObject("intro_narration_contract")
                                val orderedBeats =
                                    introNarrationContract?.optJSONArray("ordered_beats")

                                if (orderedBeats != null && orderedBeats.length() > 0) {
                                    if (introRoute.length() != orderedBeats.length() - 3) {
                                        // intro_route intentionally omits target_orientation / technique_lens / CTA;
                                        // it should still align with the middle semantic beats.
                                        problems.put(
                                            problem(
                                                "atom0_intro_route_contract_shape_mismatch",
                                                0,
                                                "advanced_setup_payload.intro_route must align with intro_narration_contract.ordered_beats"
                                            )
                                        )
                                    } else {
                                        val expectedCore = listOf(
                                            "trigger_explanation",
                                            "trigger",
                                            "bridge",
                                            "final_resolution_setup"
                                        )
                                        expectedCore.forEachIndexed { idx, expected ->
                                            if (introRoute.optString(idx) != expected) {
                                                problems.put(
                                                    problem(
                                                        "atom0_intro_route_order_mismatch_$idx",
                                                        0,
                                                        "advanced_setup_payload.intro_route[$idx] must be '$expected'"
                                                    )
                                                )
                                            }
                                        }
                                    }
                                }
                            }

                            if (introDerived.optString("target_orientation_summary").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_target_orientation_summary_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include target_orientation_summary"
                                    )
                                )
                            }

                            if (introDerived.optString("technique_lens_summary").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_technique_lens_summary_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include technique_lens_summary"
                                    )
                                )
                            }

                            if (introDerived.optString("trigger_explanation_summary").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_trigger_explanation_summary_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include trigger_explanation_summary"
                                    )
                                )
                            }

                            if (introDerived.optString("trigger_summary").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_trigger_summary_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include trigger_summary"
                                    )
                                )
                            }

                            if (introDerived.optString("bridge_summary").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_bridge_summary_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include bridge_summary"
                                    )
                                )
                            }

                            if (introDerived.optString("final_resolution_setup_summary").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_final_resolution_setup_summary_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include final_resolution_setup_summary"
                                    )
                                )
                            }

                            if (introDerived.optString("honesty_note").isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_honesty_note_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include honesty_note"
                                    )
                                )
                            }

                            val ctaKind = introDerived.optString("cta_kind")
                            if (ctaKind.isBlank()) {
                                problems.put(
                                    problem(
                                        "atom0_intro_cta_kind_missing",
                                        0,
                                        "advanced_setup_payload.intro_derived must include cta_kind"
                                    )
                                )
                            } else if (ctaKind != "SHOW_PROOF") {
                                problems.put(
                                    problem(
                                        "atom0_intro_cta_kind_invalid",
                                        0,
                                        "advanced setup atom0 must use cta_kind=SHOW_PROOF"
                                    )
                                )
                            }

                            val introNarrationContract =
                                advancedSetupPayload.optJSONObject("intro_narration_contract")
                            if (introNarrationContract != null &&
                                !introNarrationContract.optBoolean(
                                    "must_end_with_single_walkthrough_cta",
                                    false
                                )
                            ) {
                                problems.put(
                                    problem(
                                        "atom0_intro_single_walkthrough_cta_not_required",
                                        0,
                                        "advanced intro narration contract must require a single walkthrough CTA"
                                    )
                                )
                            }

                            val atomUserPrompt = atom0.optJSONObject("user_prompt")
                            val atomUserPromptCode =
                                atomUserPrompt?.optString("code").orEmpty()
                            if (atomUserPromptCode.isNotBlank() &&
                                atomUserPromptCode != "ASK_NEXT_HINT"
                            ) {
                                problems.put(
                                    problem(
                                        "atom0_user_prompt_code_not_walkthrough_native",
                                        0,
                                        "advanced intro atom0 user_prompt.code must normalize to ASK_NEXT_HINT"
                                    )
                                )
                            }
                        }

                        val surfacedTriggerPacket = atom0Args?.optJSONObject("trigger_packet")
                        if (surfacedTriggerPacket != null &&
                            !isMeaningfulJsonObjectV2(surfacedTriggerPacket)
                        ) {
                            problems.put(
                                problem(
                                    "atom0_trigger_packet_unpopulated",
                                    0,
                                    "Advanced atom0 surfaced trigger_packet must be populated when present"
                                )
                            )
                        }
                    }

                    when (atom0Archetype) {
                        NarrativeArchetypeV1.SUBSETS.wire -> {
                            if (atom0Args != null) {
                                validateAdvancedAtom0SubsetPayloadV2(atom0Args, problems)
                            }
                        }

                        NarrativeArchetypeV1.INTERSECTIONS.wire -> {
                            if (atom0Args != null) {
                                validateAdvancedAtom0IntersectionPayloadV2(atom0Args, problems)
                            }
                        }
                    }
                }
            }

            if (kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT") {
                if (!isValidHouse(finalPrimaryHouse)) {
                    problems.put(
                        problem(
                            "final_resolution_primary_house_missing",
                            null,
                            "House-based final_resolution requires valid primary_house"
                        )
                    )
                }

                if (!sameHouse(atom0PrimaryHouse, finalPrimaryHouse)) {
                    problems.put(
                        problem(
                            "atom0_primary_house_mismatch_final_resolution",
                            0,
                            "atom0 primary_house must equal final_resolution.primary_house for house-based resolutions"
                        )
                    )
                }
            }

            val lastAtom = atoms.optJSONObject((atoms.length() - 1).coerceAtLeast(0))

            val commitFocusIdx =
                lastAtom?.optJSONObject("focus")
                    ?.optJSONObject("target_cell")
                    ?.optInt("cellIndex", -1)
                    ?: -1

            val commitFocusDigit =
                lastAtom?.optJSONObject("focus")
                    ?.optInt("target_digit", -1)
                    ?: -1

            if (commitFocusIdx in 0..80 && finalFocusIdx in 0..80 && commitFocusIdx != finalFocusIdx) {
                problems.put(
                    problem(
                        "commit_focus_mismatch_final_resolution",
                        atoms.length() - 1,
                        "COMMIT focus cell must equal final_resolution.focus_cell"
                    )
                )
            }

            if (commitFocusDigit in 1..9 && finalDigit in 1..9 && commitFocusDigit != finalDigit) {
                problems.put(
                    problem(
                        "commit_digit_mismatch_final_resolution",
                        atoms.length() - 1,
                        "COMMIT focus digit must equal final_resolution.digit"
                    )
                )
            }

            val crossProblems = validateTruthAtom0ConsistencyV2(packet)
            for (i in 0 until crossProblems.length()) {
                problems.put(crossProblems.optJSONObject(i) ?: continue)
            }
        }

        private fun isValidHouse(h: JSONObject?): Boolean {
            if (h == null) return false
            val type = h.optString("type", "")
            val idx = h.optInt("index1to9", -1)
            return type in setOf("row", "col", "box") && idx in 1..9
        }

        private fun sameHouse(a: JSONObject?, b: JSONObject?): Boolean {
            if (a == null || b == null) return false
            return a.optString("type", "") == b.optString("type", "") &&
                    a.optInt("index1to9", -1) == b.optInt("index1to9", -1)
        }


        private fun verdict(ok: Boolean, atomsCount: Int, problems: JSONArray): JSONObject =
            JSONObject().apply {
                put("schema_version", "proof_verdict_v1")
                put("ok", ok)
                put("atoms_count", atomsCount)
                put("problems", problems)
                val p = problems.length().coerceAtLeast(0)
                val denom = (atomsCount.coerceAtLeast(1) * 6)
                val score = (1.0 - (p.toDouble() / denom.toDouble())).coerceIn(0.0, 1.0)
                put("proof_completeness_score", score)
            }

        private fun problem(code: String, atomIndex: Int?, note: String): JSONObject =
            JSONObject().apply {
                put("code", code)
                put("atom_index", atomIndex ?: JSONObject.NULL)
                put("note", note)
            }



        private fun validateStageShapeContract(
            atoms: JSONArray,
            problems: JSONArray
        ) {
            if (atoms.length() <= 0) return

            val atom0 = atoms.optJSONObject(0)
            val atom0Beat = atom0?.optString("beat_kind", "").orEmpty()
            if (atom0Beat != "SPOTLIGHT") {
                problems.put(
                    problem(
                        "atom0_not_spotlight",
                        0,
                        "Atom 0 must be SPOTLIGHT"
                    )
                )
            }

            if (atoms.length() >= 2) {
                val atom1 = atoms.optJSONObject(1)
                val beat1 = atom1?.optString("beat_kind", "").orEmpty()
                if (beat1 == "COMMIT") {
                    problems.put(
                        problem(
                            "atom1_commit_too_early",
                            1,
                            "Atom 1 must not be COMMIT"
                        )
                    )
                }
            }

            val last = atoms.optJSONObject((atoms.length() - 1).coerceAtLeast(0))
            val lastBeat = last?.optString("beat_kind", "").orEmpty()
            if (lastBeat != "COMMIT") {
                problems.put(
                    problem(
                        "last_atom_not_commit",
                        atoms.length() - 1,
                        "Last atom must be COMMIT"
                    )
                )
            }
        }

        private fun validateSetupAtomPurity(
            atoms: JSONArray,
            problems: JSONArray
        ) {
            val atom0 = atoms.optJSONObject(0) ?: return
            val witnesses = atom0.optJSONArray("witnesses") ?: JSONArray()
            val effects = atom0.optJSONObject("effects") ?: JSONObject()
            val eliminations = effects.optJSONArray("eliminations") ?: JSONArray()
            val placements = effects.optJSONArray("placements") ?: JSONArray()

            if (witnesses.length() > 0) {
                problems.put(
                    problem(
                        "atom0_has_witnesses",
                        0,
                        "Atom 0 must not carry witnesses"
                    )
                )
            }

            if (eliminations.length() > 0) {
                problems.put(
                    problem(
                        "atom0_has_eliminations",
                        0,
                        "Atom 0 must not carry eliminations"
                    )
                )
            }

            if (placements.length() > 0) {
                problems.put(
                    problem(
                        "atom0_has_placements",
                        0,
                        "Atom 0 must not carry placements"
                    )
                )
            }
        }


    }