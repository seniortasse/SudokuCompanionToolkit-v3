package com.contextionary.sudoku.conductor.policy

import org.json.JSONArray
import org.json.JSONObject

/**
 * Phase 5/7 projector layer adapted to the ACTUAL ReplyRequestV1 shape used
 * in this codebase:
 *
 * - replyRequest.turn.*
 * - replyRequest.decision.*
 * - replyRequest.facts: List<FactBundleV1>
 * - replyRequest.recentTurns: List<TranscriptTurnV1>
 */
object ReplySupplyProjectorsV1 {

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private fun copyIfPresent(src: JSONObject, dst: JSONObject, key: String) {
        if (src.has(key) && !src.isNull(key)) {
            dst.put(key, src.get(key))
        }
    }

    private fun truncateForMini(
        text: String?,
        maxChars: Int = 160
    ): String? {
        val cleaned = text?.trim()?.replace(Regex("\\s+"), " ").orEmpty()
        if (cleaned.isBlank()) return null
        return if (cleaned.length <= maxChars) cleaned else cleaned.take(maxChars - 1).trimEnd() + "…"
    }

    private fun takeFirstNRecentTurns(
        src: List<TranscriptTurnV1>,
        n: Int
    ): JSONArray {
        val out = JSONArray()
        if (n <= 0) return out
        src.takeLast(n).forEach { out.put(it.toJson()) }
        return out
    }

    private fun storyToJson(story: ReplyStoryCtxV1?): JSONObject {
        return story?.toJson() ?: JSONObject()
    }

    private enum class CanonicalStageAuthorityV1 {
        SETUP,
        CONFRONTATION,
        RESOLUTION,
        NON_SOLVING
    }

    private fun canonicalStageAuthorityV1(
        replyRequest: ReplyRequestV1
    ): CanonicalStageAuthorityV1 {
        val story = replyRequest.turn.story
        val canonicalKind = story?.canonicalPositionKind?.trim()?.uppercase()
        val stage = story?.stage?.trim()?.uppercase()

        return when {
            canonicalKind == "SETUP" || stage == "SETUP" ->
                CanonicalStageAuthorityV1.SETUP

            canonicalKind == "CONFRONTATION" || stage == "CONFRONTATION" ->
                CanonicalStageAuthorityV1.CONFRONTATION

            canonicalKind == "RESOLUTION_COMMIT" ||
                    canonicalKind == "RESOLUTION_POST_COMMIT" ||
                    stage == "RESOLUTION" ->
                CanonicalStageAuthorityV1.RESOLUTION

            else ->
                CanonicalStageAuthorityV1.NON_SOLVING
        }
    }

    private fun stageAllowsPacketV1(
        replyRequest: ReplyRequestV1,
        packetType: FactBundleV1.Type
    ): Boolean {
        return when (canonicalStageAuthorityV1(replyRequest)) {
            CanonicalStageAuthorityV1.SETUP ->
                packetType == FactBundleV1.Type.SETUP_REPLY_PACKET_V1

            CanonicalStageAuthorityV1.CONFRONTATION ->
                packetType == FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1

            CanonicalStageAuthorityV1.RESOLUTION ->
                packetType == FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1

            CanonicalStageAuthorityV1.NON_SOLVING ->
                false
        }
    }

    private fun findAuthorizedFactPayload(
        replyRequest: ReplyRequestV1,
        packetType: FactBundleV1.Type
    ): JSONObject? {
        if (!stageAllowsPacketV1(replyRequest, packetType)) return null
        val match = replyRequest.facts.firstOrNull { it.type == packetType }
        return match?.payload
    }

    private fun resolutionStageAuthorizedV1(
        replyRequest: ReplyRequestV1
    ): Boolean =
        canonicalStageAuthorityV1(replyRequest) == CanonicalStageAuthorityV1.RESOLUTION

    private fun findFactPayload(
        replyRequest: ReplyRequestV1,
        typeName: String
    ): JSONObject? {
        val match = replyRequest.facts.firstOrNull { it.type.name == typeName }
        return match?.payload
    }

    private fun collectFacts(
        replyRequest: ReplyRequestV1,
        typeNames: Set<String>
    ): JSONArray {
        val out = JSONArray()
        replyRequest.facts.forEach { fact ->
            if (typeNames.contains(fact.type.name)) {
                out.put(
                    JSONObject().apply {
                        put("type", fact.type.name)
                        put("payload", fact.payload)
                    }
                )
            }
        }
        return out
    }

    private fun projectStoryHeaderMini(story: ReplyStoryCtxV1?): JSONObject {
        val src = storyToJson(story)
        val out = JSONObject()
        copyIfPresent(src, out, "stage")
        copyIfPresent(src, out, "step_id")
        copyIfPresent(src, out, "ready_for_commit")
        copyIfPresent(src, out, "canonical_pending_kind")
        return out
    }

    // ---------------------------------------------------------------------
    // Global compact projectors
    // ---------------------------------------------------------------------

    fun projectTurnHeaderMini(replyRequest: ReplyRequestV1): JSONObject {
        val t = replyRequest.turn
        return JSONObject().apply {
            put("turn_id", t.turnId)
            put("phase", t.phase)
            put("mode", t.mode)
            put("pending_after", t.pendingAfter ?: JSONObject.NULL)

            val storyMini = projectStoryHeaderMini(t.story)
            if (storyMini.length() > 0) {
                put("story", storyMini)
            }
        }
    }

    fun projectStyleMini(replyRequest: ReplyRequestV1): JSONObject {
        return replyRequest.style.toJson()
    }

    fun projectDecisionSummaryMini(replyRequest: ReplyRequestV1): JSONObject {
        val decision = replyRequest.decision
        val allowMutationSurface = resolutionStageAuthorizedV1(replyRequest)
        val visibleMutation =
            if (allowMutationSurface) decision.mutationApplied else null

        return JSONObject().apply {
            put("decision_kind", decision.decisionKind.name)
            put(
                "decision_hint",
                truncateForMini(decision.summary, maxChars = 140) ?: JSONObject.NULL
            )
            put(
                "has_mutation_applied",
                visibleMutation != null
            )
            put(
                "mutation_applied",
                visibleMutation?.toJson() ?: JSONObject.NULL
            )
            put(
                "mutation_stage_gate",
                if (allowMutationSurface) "RESOLUTION_AUTHORIZED" else "SUPPRESSED_NON_RESOLUTION_STAGE"
            )
        }
    }

    fun projectContinuityShort(replyRequest: ReplyRequestV1): JSONObject {
        val isSetupTurn = replyRequest.turn.story?.stage == "SETUP"

        val userTally = replyRequest.userTally
        val assistantTally = replyRequest.assistantTally
        val handover = findFactPayload(replyRequest, FactBundleV1.Type.HANDOVER_NOTE_V1.name)
        val returnToRoute = findFactPayload(replyRequest, FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1.name)

        val authorityOwner = replyRequest.turn.turnAuthorityOwner.orEmpty()
        val isDetourLikeOwner =
            authorityOwner == "USER_DETOUR_OWNER" || authorityOwner == "REPAIR_OWNER"
        val isRouteJumpOwner =
            authorityOwner == "USER_ROUTE_JUMP_OWNER"

        val continuityLane =
            when {
                isDetourLikeOwner -> "user_detour"
                isRouteJumpOwner -> "route_jump"
                returnToRoute != null -> "app_route"
                else -> "app_route"
            }

        val out = JSONObject().apply {
            put("continuity_lane", continuityLane)
        }

        val userMini = JSONObject().apply {
            userTally?.name?.let { put("name", it) }
            if (isDetourLikeOwner) {
                userTally?.age?.let { put("age", it) }
                userTally?.preferences?.takeIf { it.isNotBlank() }?.let { put("preferences", it) }
                userTally?.dislikes?.takeIf { it.isNotBlank() }?.let { put("dislikes", it) }
                userTally?.personality?.takeIf { it.isNotBlank() }?.let { put("personality", it) }
                userTally?.facts?.takeIf { it.isNotBlank() }?.let { put("facts", it) }
                userTally?.sudokuLevel?.takeIf { it.isNotBlank() }?.let { put("sudoku_level", it) }
                userTally?.thinkingProcess?.takeIf { it.isNotBlank() }?.let { put("thinking_process", it) }
                userTally?.firstSpeech?.takeIf { it.isNotBlank() }?.let { put("first_speech", it) }
            }
        }
        if (userMini.length() > 0) {
            out.put("user_tally", userMini)
        }

        val assistantMini = JSONObject().apply {
            assistantTally?.name?.let { put("name", it) }
            assistantTally?.personality?.let { put("personality", it) }
            if (isDetourLikeOwner) {
                assistantTally?.age?.let { put("age", it) }
                assistantTally?.about?.takeIf { it.isNotBlank() }?.let { put("about", it) }
                assistantTally?.preferences?.takeIf { it.isNotBlank() }?.let { put("preferences", it) }
                assistantTally?.dislikes?.takeIf { it.isNotBlank() }?.let { put("dislikes", it) }
                assistantTally?.firstSpeech?.takeIf { it.isNotBlank() }?.let { put("first_speech", it) }
            }
        }
        if (assistantMini.length() > 0) {
            out.put("assistant_tally", assistantMini)
        }

        val bridgeHints = handover?.optJSONArray("bridge_hints") ?: JSONArray()
        if (bridgeHints.length() > 0) {
            out.put("transition_hint", bridgeHints.opt(0))
        }

        if (returnToRoute != null) {
            val resumeMini = JSONObject().apply {
                put("phase", returnToRoute.optString("phase"))
                put("app_agenda_kind", returnToRoute.opt("app_agenda_kind") ?: JSONObject.NULL)
                put("step_id", returnToRoute.opt("step_id") ?: JSONObject.NULL)
                put("story_stage", returnToRoute.opt("story_stage") ?: JSONObject.NULL)
                put("resume_prompt_hint", returnToRoute.opt("resume_prompt_hint") ?: JSONObject.NULL)
                put("resume_reason", returnToRoute.opt("resume_reason") ?: JSONObject.NULL)
            }
            out.put("resume_anchor", resumeMini)
        }

        val recentLimit =
            when {
                isDetourLikeOwner -> 3
                isRouteJumpOwner -> 3
                isSetupTurn -> 2
                else -> 3
            }

        val recentTurns = takeFirstNRecentTurns(replyRequest.recentTurns, recentLimit)
        if (recentTurns.length() > 0) {
            out.put("recent_turns", recentTurns)
        }

        return out
    }

    private enum class PersonalizationProjectionMode {
        SOLVING_SETUP,
        SOLVING_CONFRONTATION,
        SOLVING_RESOLUTION,
        VALIDATION,
        SOCIAL,
        MINIMAL
    }

    private data class CurrentTechniqueContextV1(
        val sourcePacket: String,
        val sourceStage: String,
        val name: String,
        val realName: String? = null,
        val family: String? = null,
        val difficultyLevel: String? = null
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("source_packet", sourcePacket)
            put("source_stage", sourceStage)
            put("name", name)
            put("real_name", realName ?: JSONObject.NULL)
            put("family", family ?: JSONObject.NULL)
            put("difficulty_level", difficultyLevel ?: JSONObject.NULL)
        }
    }

    private fun normalizedTechniqueNameFrom(src: JSONObject): String? {
        val raw =
            src.optString("app_name", "")
                .ifBlank { src.optString("spoken_technique_name", "") }
                .ifBlank { src.optString("real_name", "") }
                .ifBlank { src.optString("name", "") }
                .ifBlank { src.optString("technique_name", "") }
                .ifBlank { src.optString("id", "") }
                .trim()
        return raw.takeIf { it.isNotEmpty() }
    }

    private fun extractTechniqueContextFromPacket(
        payload: JSONObject?,
        sourcePacket: String,
        sourceStage: String
    ): CurrentTechniqueContextV1? {
        if (payload == null) return null

        val technique = payload.optJSONObject("technique") ?: return null
        val name = normalizedTechniqueNameFrom(technique) ?: return null

        val realName =
            technique.optString("real_name", "")
                .trim()
                .takeIf { it.isNotEmpty() && !it.equals(name, ignoreCase = true) }

        val family =
            technique.optString("family", "")
                .trim()
                .takeIf { it.isNotEmpty() }

        val difficultyLevel =
            technique.optString("difficulty_level", "")
                .trim()
                .takeIf { it.isNotEmpty() }

        return CurrentTechniqueContextV1(
            sourcePacket = sourcePacket,
            sourceStage = sourceStage,
            name = name,
            realName = realName,
            family = family,
            difficultyLevel = difficultyLevel
        )
    }

    private fun detectCurrentTechniqueContext(
        replyRequest: ReplyRequestV1
    ): CurrentTechniqueContextV1? {
        val setupPacket =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.SETUP_REPLY_PACKET_V1)
        val confrontationPacket =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1)
        val resolutionPacket =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1)

        return when {
            setupPacket != null ->
                extractTechniqueContextFromPacket(
                    payload = setupPacket,
                    sourcePacket = FactBundleV1.Type.SETUP_REPLY_PACKET_V1.name,
                    sourceStage = "SETUP"
                )

            confrontationPacket != null ->
                extractTechniqueContextFromPacket(
                    payload = confrontationPacket,
                    sourcePacket = FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1.name,
                    sourceStage = "CONFRONTATION"
                )

            resolutionPacket != null ->
                extractTechniqueContextFromPacket(
                    payload = resolutionPacket,
                    sourcePacket = FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1.name,
                    sourceStage = "RESOLUTION"
                )

            else -> null
        }
    }

    private data class CurrentTechniqueUserFitV1(
        val currentFamiliarity: String,
        val isRecentlyLearned: Boolean = false,
        val isChallenging: Boolean = false,
        val isFragile: Boolean = false,
        val isMastered: Boolean = false,
        val isFavorite: Boolean = false,
        val shouldExplainGently: Boolean = false,
        val shouldShortenIntro: Boolean = false,
        val shouldKeepProofExplicit: Boolean = false,
        val shouldReinforceLearning: Boolean = false,
        val shouldAvoidJargonSpike: Boolean = false,
        val matchedFrom: List<String> = emptyList(),
        val comfortNotes: String? = null
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("current_familiarity", currentFamiliarity)
            put("is_recently_learned", isRecentlyLearned)
            put("is_challenging", isChallenging)
            put("is_fragile", isFragile)
            put("is_mastered", isMastered)
            put("is_favorite", isFavorite)
            put("should_explain_gently", shouldExplainGently)
            put("should_shorten_intro", shouldShortenIntro)
            put("should_keep_proof_explicit", shouldKeepProofExplicit)
            put("should_reinforce_learning", shouldReinforceLearning)
            put("should_avoid_jargon_spike", shouldAvoidJargonSpike)
            if (matchedFrom.isNotEmpty()) {
                put("matched_from", JSONArray().apply { matchedFrom.forEach { put(it) } })
            }
            put("comfort_notes", comfortNotes ?: JSONObject.NULL)
        }
    }

    private fun techniqueNamesEquivalent(a: String?, b: String?): Boolean {
        val aa = a?.trim()?.lowercase()?.takeIf { it.isNotEmpty() } ?: return false
        val bb = b?.trim()?.lowercase()?.takeIf { it.isNotEmpty() } ?: return false
        return aa == bb
    }

    private fun containsTechniqueName(xs: List<String>, target: String?): Boolean {
        val t = target?.trim()?.takeIf { it.isNotEmpty() } ?: return false
        return xs.any { techniqueNamesEquivalent(it, t) }
    }

    private fun findTechniqueComfortEntry(
        knowledge: SudokuKnowledgeTechniqueMapV1,
        techniqueName: String
    ): TechniqueComfortEntryV1? {
        return knowledge.techniqueComfort.firstOrNull { entry ->
            techniqueNamesEquivalent(entry.technique, techniqueName)
        }
    }

    private fun deriveCurrentTechniqueUserFit(
        replyRequest: ReplyRequestV1,
        currentTechnique: CurrentTechniqueContextV1?
    ): CurrentTechniqueUserFitV1? {
        val technique = currentTechnique ?: return null
        val knowledge = replyRequest.relationshipMemory?.sudokuKnowledgeTechniqueMap ?: return null

        val comfortEntry = findTechniqueComfortEntry(knowledge, technique.name)
        val matchedFrom = mutableListOf<String>()

        val comfortFamiliarity = comfortEntry?.familiarity?.name?.lowercase()

        val isRecentlyLearned = containsTechniqueName(knowledge.recentlyLearnedTechniques, technique.name)
            .also { if (it) matchedFrom += "recently_learned_techniques" }

        val isChallenging = containsTechniqueName(knowledge.challengingTechniques, technique.name)
            .also { if (it) matchedFrom += "challenging_techniques" }

        val isFragile = containsTechniqueName(knowledge.fragileTechniques, technique.name)
            .also { if (it) matchedFrom += "fragile_techniques" }

        val isMastered = containsTechniqueName(knowledge.masteredPatterns, technique.name)
            .also { if (it) matchedFrom += "mastered_patterns" }

        val isFavorite = containsTechniqueName(knowledge.favoriteTechniques, technique.name)
            .also { if (it) matchedFrom += "favorite_techniques" }

        if (comfortEntry != null) matchedFrom += "technique_comfort"

        val currentFamiliarity =
            when {
                isMastered -> "mastered"
                isFragile -> "fragile"
                isChallenging -> "challenging"
                isRecentlyLearned -> "newly_learned"
                !comfortFamiliarity.isNullOrBlank() -> comfortFamiliarity
                else -> "unknown"
            }

        val shouldExplainGently =
            currentFamiliarity == "newly_learned" ||
                    currentFamiliarity == "fragile" ||
                    currentFamiliarity == "challenging"

        val shouldShortenIntro =
            currentFamiliarity == "mastered" ||
                    currentFamiliarity == "easy"

        val shouldKeepProofExplicit =
            currentFamiliarity == "newly_learned" ||
                    currentFamiliarity == "fragile" ||
                    currentFamiliarity == "challenging" ||
                    currentFamiliarity == "medium"

        val shouldReinforceLearning =
            currentFamiliarity == "newly_learned" ||
                    currentFamiliarity == "fragile" ||
                    currentFamiliarity == "medium"

        val shouldAvoidJargonSpike =
            currentFamiliarity == "newly_learned" ||
                    currentFamiliarity == "fragile" ||
                    currentFamiliarity == "challenging"

        return CurrentTechniqueUserFitV1(
            currentFamiliarity = currentFamiliarity,
            isRecentlyLearned = isRecentlyLearned,
            isChallenging = isChallenging,
            isFragile = isFragile,
            isMastered = isMastered,
            isFavorite = isFavorite,
            shouldExplainGently = shouldExplainGently,
            shouldShortenIntro = shouldShortenIntro,
            shouldKeepProofExplicit = shouldKeepProofExplicit,
            shouldReinforceLearning = shouldReinforceLearning,
            shouldAvoidJargonSpike = shouldAvoidJargonSpike,
            matchedFrom = matchedFrom.distinct(),
            comfortNotes = comfortEntry?.notes
        )
    }

    private fun detectPersonalizationProjectionMode(
        replyRequest: ReplyRequestV1
    ): PersonalizationProjectionMode {
        val mode = replyRequest.turn.mode.trim().uppercase()
        val phase = replyRequest.turn.phase.trim().uppercase()

        when (canonicalStageAuthorityV1(replyRequest)) {
            CanonicalStageAuthorityV1.SETUP ->
                return PersonalizationProjectionMode.SOLVING_SETUP

            CanonicalStageAuthorityV1.CONFRONTATION ->
                return PersonalizationProjectionMode.SOLVING_CONFRONTATION

            CanonicalStageAuthorityV1.RESOLUTION ->
                return PersonalizationProjectionMode.SOLVING_RESOLUTION

            CanonicalStageAuthorityV1.NON_SOLVING -> Unit
        }

        if (replyRequest.openingTurn || mode == "FREE_TALK") {
            return PersonalizationProjectionMode.SOCIAL
        }

        val hasConfirmingPacket =
            findFactPayload(replyRequest, FactBundleV1.Type.CONFIRMING_RETAKE_PACKET.name) != null ||
                    findFactPayload(replyRequest, FactBundleV1.Type.CONFIRMING_MISMATCH_PACKET.name) != null ||
                    findFactPayload(replyRequest, FactBundleV1.Type.CONFIRMING_CONFLICT_PACKET.name) != null ||
                    findFactPayload(replyRequest, FactBundleV1.Type.CONFIRMING_VISUAL_VERIFY_PACKET.name) != null ||
                    findFactPayload(replyRequest, FactBundleV1.Type.CONFIRMING_NOT_UNIQUE_PACKET.name) != null ||
                    findFactPayload(replyRequest, FactBundleV1.Type.CONFIRMING_FINALIZE_PACKET.name) != null

        val hasPendingContext =
            findFactPayload(replyRequest, FactBundleV1.Type.PENDING_CONTEXT_V1.name) != null

        if (phase == "CONFIRMING" || hasConfirmingPacket || hasPendingContext) {
            return PersonalizationProjectionMode.VALIDATION
        }

        return PersonalizationProjectionMode.MINIMAL
    }

    private fun buildSpeechStyleJson(
        replyRequest: ReplyRequestV1,
        includePace: Boolean = true,
        includeRhythm: Boolean = true,
        includeVerbosity: Boolean = true,
        includeConfirmationStyle: Boolean = true
    ): JSONObject {
        val speech = replyRequest.relationshipMemory?.communicationSpeechStyle ?: return JSONObject()
        return JSONObject().apply {
            if (includePace) speech.pacePreference?.let { put("pace", it.name.lowercase()) }
            if (includeRhythm) speech.speechRhythmPreference?.let { put("rhythm", it) }
            if (includeVerbosity) speech.verbosityPreference?.let { put("verbosity", it) }
            if (includeConfirmationStyle) {
                speech.confirmationStylePreference?.let { put("confirmation_style", it) }
            }
        }
    }

    private fun buildLearningBiasJson(
        replyRequest: ReplyRequestV1,
        includePreferences: Boolean = true,
        includeProofPreference: Boolean = true,
        includeSetupPreference: Boolean = true,
        includeConfrontationPreference: Boolean = true,
        includeResolutionPreference: Boolean = true
    ): JSONObject {
        val learning = replyRequest.relationshipMemory?.learningExplanationModel ?: return JSONObject()
        return JSONObject().apply {
            if (includePreferences && learning.learningPreferences.isNotEmpty()) {
                put(
                    "preferences",
                    JSONArray().apply {
                        learning.learningPreferences.forEach { put(it.name.lowercase()) }
                    }
                )
            }
            if (includeProofPreference) {
                learning.proofPreference?.let { put("proof_preference", it.name.lowercase()) }
            }
            if (includeSetupPreference) {
                learning.setupPreference?.let { put("setup_preference", it) }
            }
            if (includeConfrontationPreference) {
                learning.confrontationPreference?.let { put("confrontation_preference", it) }
            }
            if (includeResolutionPreference) {
                learning.resolutionPreference?.let { put("resolution_preference", it) }
            }
        }
    }

    private fun buildTechniqueContextJson(
        replyRequest: ReplyRequestV1,
        includeCurrentTechnique: Boolean = true,
        includeCurrentUserFit: Boolean = true,
        includeComfort: Boolean = true,
        includeRecentlyLearned: Boolean = true,
        includeChallenging: Boolean = true
    ): JSONObject {
        val knowledge = replyRequest.relationshipMemory?.sudokuKnowledgeTechniqueMap ?: return JSONObject()
        val currentTechnique = if (includeCurrentTechnique) detectCurrentTechniqueContext(replyRequest) else null
        val currentUserFit =
            if (includeCurrentUserFit) deriveCurrentTechniqueUserFit(replyRequest, currentTechnique) else null

        return JSONObject().apply {
            currentTechnique?.let { put("current_technique", it.toJson()) }
            currentUserFit?.let { put("current_user_fit", it.toJson()) }

            if (includeComfort && knowledge.techniqueComfort.isNotEmpty()) {
                put(
                    "technique_comfort",
                    JSONArray().apply {
                        knowledge.techniqueComfort.take(8).forEach { entry ->
                            put(
                                JSONObject().apply {
                                    put("technique", entry.technique)
                                    put("familiarity", entry.familiarity.name.lowercase())
                                    put("notes", entry.notes ?: JSONObject.NULL)
                                }
                            )
                        }
                    }
                )
            }
            if (includeRecentlyLearned && knowledge.recentlyLearnedTechniques.isNotEmpty()) {
                put(
                    "recently_learned_techniques",
                    JSONArray().apply {
                        knowledge.recentlyLearnedTechniques.take(8).forEach { put(it) }
                    }
                )
            }
            if (includeChallenging && knowledge.challengingTechniques.isNotEmpty()) {
                put(
                    "challenging_techniques",
                    JSONArray().apply {
                        knowledge.challengingTechniques.take(8).forEach { put(it) }
                    }
                )
            }
        }
    }

    private fun buildAvoidJson(
        replyRequest: ReplyRequestV1,
        includeSpeechAvoid: Boolean = true,
        includeConfusionTriggers: Boolean = true,
        includeRecentFriction: Boolean = true
    ): JSONArray {
        val memory = replyRequest.relationshipMemory ?: return JSONArray()
        val speech = memory.communicationSpeechStyle
        val learning = memory.learningExplanationModel
        val integrity = memory.interactionHistoryMemoryIntegrity

        val out = JSONArray()
        if (includeSpeechAvoid) speech.avoidSpeechPatterns.take(8).forEach { out.put(it) }
        if (includeConfusionTriggers) learning.confusionTriggers.take(6).forEach { out.put(it) }
        if (includeRecentFriction) integrity.recentFrictionEdges.take(6).forEach { out.put(it) }
        return out
    }

    private fun buildUserTermsJson(
        replyRequest: ReplyRequestV1,
        includeUserJargon: Boolean = true,
        includePreferredTerms: Boolean = true
    ): JSONArray {
        val language = replyRequest.relationshipMemory?.personalLanguageMeaningHooks ?: return JSONArray()
        val out = JSONArray()
        if (includeUserJargon) language.userJargon.take(8).forEach { out.put(it) }
        if (includePreferredTerms) language.preferredTerms.take(8).forEach { out.put(it) }
        return out
    }

    private data class RankedPersonalizationFieldV1(
        val key: String,
        val score: Int,
        val value: Any
    )

    private fun addRankedField(
        xs: MutableList<RankedPersonalizationFieldV1>,
        key: String,
        score: Int,
        value: Any?
    ) {
        if (value == null) return
        when (value) {
            is JSONObject -> if (value.length() == 0) return
            is JSONArray -> if (value.length() == 0) return
            is String -> if (value.isBlank()) return
        }
        xs += RankedPersonalizationFieldV1(key = key, score = score, value = value)
    }

    private fun maxProjectedFieldCount(mode: PersonalizationProjectionMode): Int {
        return when (mode) {
            PersonalizationProjectionMode.SOLVING_SETUP -> 6
            PersonalizationProjectionMode.SOLVING_CONFRONTATION -> 6
            PersonalizationProjectionMode.SOLVING_RESOLUTION -> 5
            PersonalizationProjectionMode.VALIDATION -> 4
            PersonalizationProjectionMode.SOCIAL -> 5
            PersonalizationProjectionMode.MINIMAL -> 3
        }
    }

    private fun scoreRelationshipTone(
        mode: PersonalizationProjectionMode
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP -> 80
        PersonalizationProjectionMode.SOLVING_CONFRONTATION -> 40
        PersonalizationProjectionMode.SOLVING_RESOLUTION -> 78
        PersonalizationProjectionMode.VALIDATION -> 45
        PersonalizationProjectionMode.SOCIAL -> 95
        PersonalizationProjectionMode.MINIMAL -> 5
    }

    private fun scoreSpeechStyle(
        mode: PersonalizationProjectionMode
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP -> 92
        PersonalizationProjectionMode.SOLVING_CONFRONTATION -> 94
        PersonalizationProjectionMode.SOLVING_RESOLUTION -> 74
        PersonalizationProjectionMode.VALIDATION -> 96
        PersonalizationProjectionMode.SOCIAL -> 72
        PersonalizationProjectionMode.MINIMAL -> 98
    }

    private fun scoreLearningBias(
        mode: PersonalizationProjectionMode,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP ->
            if (currentUserFit?.shouldExplainGently == true) 95 else 88

        PersonalizationProjectionMode.SOLVING_CONFRONTATION ->
            if (currentUserFit?.shouldKeepProofExplicit == true) 100 else 90

        PersonalizationProjectionMode.SOLVING_RESOLUTION -> 55
        PersonalizationProjectionMode.VALIDATION -> 10
        PersonalizationProjectionMode.SOCIAL -> 15
        PersonalizationProjectionMode.MINIMAL -> 0
    }

    private fun scoreTechniqueContext(
        mode: PersonalizationProjectionMode,
        currentTechnique: CurrentTechniqueContextV1?,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): Int {
        if (currentTechnique == null) return 0
        return when (mode) {
            PersonalizationProjectionMode.SOLVING_SETUP ->
                if (currentUserFit != null) 96 else 82

            PersonalizationProjectionMode.SOLVING_CONFRONTATION ->
                if (currentUserFit?.shouldKeepProofExplicit == true) 99 else 92

            PersonalizationProjectionMode.SOLVING_RESOLUTION ->
                if (currentUserFit?.shouldReinforceLearning == true) 88 else 72

            PersonalizationProjectionMode.VALIDATION -> 0
            PersonalizationProjectionMode.SOCIAL -> 0
            PersonalizationProjectionMode.MINIMAL -> 0
        }
    }

    private fun scoreMetaphorPolicy(
        mode: PersonalizationProjectionMode,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP ->
            if (currentUserFit?.shouldExplainGently == true) 58 else 68

        PersonalizationProjectionMode.SOLVING_CONFRONTATION ->
            if (currentUserFit?.shouldKeepProofExplicit == true) 15 else 35

        PersonalizationProjectionMode.SOLVING_RESOLUTION -> 28
        PersonalizationProjectionMode.VALIDATION -> 0
        PersonalizationProjectionMode.SOCIAL -> 82
        PersonalizationProjectionMode.MINIMAL -> 0
    }

    private fun scoreAvoid(
        mode: PersonalizationProjectionMode,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP ->
            if (currentUserFit?.shouldAvoidJargonSpike == true) 90 else 76

        PersonalizationProjectionMode.SOLVING_CONFRONTATION ->
            if (currentUserFit?.shouldAvoidJargonSpike == true) 97 else 85

        PersonalizationProjectionMode.SOLVING_RESOLUTION -> 62
        PersonalizationProjectionMode.VALIDATION -> 92
        PersonalizationProjectionMode.SOCIAL -> 35
        PersonalizationProjectionMode.MINIMAL -> 78
    }

    private fun scoreUserTerms(
        mode: PersonalizationProjectionMode
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP -> 72
        PersonalizationProjectionMode.SOLVING_CONFRONTATION -> 74
        PersonalizationProjectionMode.SOLVING_RESOLUTION -> 48
        PersonalizationProjectionMode.VALIDATION -> 76
        PersonalizationProjectionMode.SOCIAL -> 86
        PersonalizationProjectionMode.MINIMAL -> 82
    }

    private fun scoreCoachingNote(
        mode: PersonalizationProjectionMode,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): Int = when (mode) {
        PersonalizationProjectionMode.SOLVING_SETUP ->
            if (currentUserFit != null) 70 else 58

        PersonalizationProjectionMode.SOLVING_CONFRONTATION ->
            if (currentUserFit?.shouldKeepProofExplicit == true) 80 else 60

        PersonalizationProjectionMode.SOLVING_RESOLUTION ->
            if (currentUserFit?.shouldReinforceLearning == true) 92 else 74

        PersonalizationProjectionMode.VALIDATION -> 66
        PersonalizationProjectionMode.SOCIAL -> 84
        PersonalizationProjectionMode.MINIMAL -> 20
    }

    private fun buildSetupCoachingNote(
        replyRequest: ReplyRequestV1,
        currentTechnique: CurrentTechniqueContextV1?,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): String? {
        val setupPreference =
            replyRequest.relationshipMemory?.learningExplanationModel?.setupPreference?.trim()
                ?.takeIf { it.isNotEmpty() }

        return when {
            currentUserFit?.shouldShortenIntro == true && currentTechnique != null ->
                "Keep the setup short; ${currentTechnique.name} already seems familiar to the user."

            currentUserFit?.shouldExplainGently == true && currentTechnique != null ->
                "Frame ${currentTechnique.name} gently and make the entry feel approachable."

            !setupPreference.isNullOrBlank() ->
                "Follow the user's preferred setup style: $setupPreference."

            currentTechnique != null ->
                "Set up ${currentTechnique.name} cleanly and get to the key structure without padding."

            else ->
                "Keep the setup focused and get to the key structure without padding."
        }
    }

    private fun buildConfrontationCoachingNote(
        replyRequest: ReplyRequestV1,
        currentTechnique: CurrentTechniqueContextV1?,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): String? {
        val confrontationPreference =
            replyRequest.relationshipMemory?.learningExplanationModel?.confrontationPreference?.trim()
                ?.takeIf { it.isNotEmpty() }

        return when {
            currentUserFit?.shouldKeepProofExplicit == true && currentTechnique != null ->
                "Keep the ${currentTechnique.name} proof explicit and avoid abstraction spikes."

            currentUserFit?.shouldAvoidJargonSpike == true && currentTechnique != null ->
                "Explain ${currentTechnique.name} with visible eliminations and restrained jargon."

            !confrontationPreference.isNullOrBlank() ->
                "Shape the confrontation around the user's preferred proof style: $confrontationPreference."

            currentTechnique != null ->
                "Keep the ${currentTechnique.name} reasoning visible step by step."

            else ->
                "Keep the reasoning visible step by step."
        }
    }

    private fun buildResolutionCoachingNote(
        replyRequest: ReplyRequestV1,
        currentTechnique: CurrentTechniqueContextV1?,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): String? {
        val resolutionPreference =
            replyRequest.relationshipMemory?.learningExplanationModel?.resolutionPreference?.trim()
                ?.takeIf { it.isNotEmpty() }

        return when {
            currentUserFit?.shouldReinforceLearning == true && currentTechnique != null ->
                "Use the close to reinforce ${currentTechnique.name} without turning it into a long moral."

            currentUserFit?.isMastered == true && currentTechnique != null ->
                "Keep the close compact; ${currentTechnique.name} seems well established for this user."

            !resolutionPreference.isNullOrBlank() ->
                "Match the close to the user's preferred resolution style: $resolutionPreference."

            currentTechnique != null ->
                "Close ${currentTechnique.name} with a brief, confidence-building takeaway."

            else ->
                "Close with a brief, confidence-building takeaway."
        }
    }

    private fun buildValidationCoachingNote(
        replyRequest: ReplyRequestV1
    ): String? {
        val confirmationStyle =
            replyRequest.relationshipMemory?.communicationSpeechStyle?.confirmationStylePreference?.trim()
                ?.takeIf { it.isNotEmpty() }

        return when {
            !confirmationStyle.isNullOrBlank() ->
                "Keep this exact and tactful, using the user's preferred confirmation style: $confirmationStyle."

            else ->
                "Keep this exact, tactful, and free of patronizing padding."
        }
    }

    private fun buildSocialCoachingNote(
        replyRequest: ReplyRequestV1
    ): String? {
        val toneBond = replyRequest.relationshipMemory?.relationshipToneBond
        val familiarity = toneBond?.familiarityPreference?.name?.lowercase()
        val humor = toneBond?.humorPreference?.name?.lowercase()

        return when {
            !familiarity.isNullOrBlank() && !humor.isNullOrBlank() ->
                "Let the social tone feel $familiarity with $humor humor, but keep it natural."

            !familiarity.isNullOrBlank() ->
                "Let the social tone feel $familiarity, but keep it natural."

            !humor.isNullOrBlank() ->
                "Use a socially natural tone with $humor humor if it fits."

            else ->
                "Let the warmth feel natural and observed, not scripted."
        }
    }

    private fun buildMinimalCoachingNote(): String? = null

    private fun buildTurnSpecificCoachingNote(
        replyRequest: ReplyRequestV1,
        mode: PersonalizationProjectionMode,
        currentTechnique: CurrentTechniqueContextV1?,
        currentUserFit: CurrentTechniqueUserFitV1?
    ): String? {
        return when (mode) {
            PersonalizationProjectionMode.SOLVING_SETUP ->
                buildSetupCoachingNote(replyRequest, currentTechnique, currentUserFit)

            PersonalizationProjectionMode.SOLVING_CONFRONTATION ->
                buildConfrontationCoachingNote(replyRequest, currentTechnique, currentUserFit)

            PersonalizationProjectionMode.SOLVING_RESOLUTION ->
                buildResolutionCoachingNote(replyRequest, currentTechnique, currentUserFit)

            PersonalizationProjectionMode.VALIDATION ->
                buildValidationCoachingNote(replyRequest)

            PersonalizationProjectionMode.SOCIAL ->
                buildSocialCoachingNote(replyRequest)

            PersonalizationProjectionMode.MINIMAL ->
                buildMinimalCoachingNote()
        }
    }

    private fun projectRankedFields(
        out: JSONObject,
        mode: PersonalizationProjectionMode,
        candidates: List<RankedPersonalizationFieldV1>
    ) {
        candidates
            .sortedWith(compareByDescending<RankedPersonalizationFieldV1> { it.score }.thenBy { it.key })
            .take(maxProjectedFieldCount(mode))
            .forEach { field ->
                out.put(field.key, field.value)
            }
    }

    fun projectPersonalizationMini(replyRequest: ReplyRequestV1): JSONObject {
        val memory = replyRequest.relationshipMemory ?: return JSONObject()

        val toneBond = memory.relationshipToneBond
        val learning = memory.learningExplanationModel
        val integrity = memory.interactionHistoryMemoryIntegrity
        val mode = detectPersonalizationProjectionMode(replyRequest)
        val currentTechnique = detectCurrentTechniqueContext(replyRequest)
        val currentUserFit = deriveCurrentTechniqueUserFit(replyRequest, currentTechnique)
        val turnSpecificCoachingNote =
            buildTurnSpecificCoachingNote(
                replyRequest = replyRequest,
                mode = mode,
                currentTechnique = currentTechnique,
                currentUserFit = currentUserFit
            )

        val out = JSONObject().apply {
            put("projection_mode", mode.name.lowercase())
        }

        val candidates = mutableListOf<RankedPersonalizationFieldV1>()

        when (mode) {
            PersonalizationProjectionMode.SOLVING_SETUP -> {
                addRankedField(
                    candidates,
                    "relationship_tone",
                    scoreRelationshipTone(mode),
                    toneBond.relationshipTone?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "speech_style",
                    scoreSpeechStyle(mode),
                    buildSpeechStyleJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "learning_bias",
                    scoreLearningBias(mode, currentUserFit),
                    buildLearningBiasJson(
                        replyRequest,
                        includePreferences = true,
                        includeProofPreference = true,
                        includeSetupPreference = true,
                        includeConfrontationPreference = false,
                        includeResolutionPreference = false
                    )
                )

                addRankedField(
                    candidates,
                    "technique_context",
                    scoreTechniqueContext(mode, currentTechnique, currentUserFit),
                    buildTechniqueContextJson(
                        replyRequest,
                        includeCurrentTechnique = true,
                        includeCurrentUserFit = true,
                        includeComfort = true,
                        includeRecentlyLearned = true,
                        includeChallenging = true
                    )
                )

                val metaphorPolicy = learning.metaphorPolicy
                addRankedField(
                    candidates,
                    "metaphor_policy",
                    scoreMetaphorPolicy(mode, currentUserFit),
                    if (metaphorPolicy.allowed || metaphorPolicy.domains.isNotEmpty() || !metaphorPolicy.frequency.isNullOrBlank()) {
                        metaphorPolicy.toJson()
                    } else null
                )

                addRankedField(
                    candidates,
                    "avoid",
                    scoreAvoid(mode, currentUserFit),
                    buildAvoidJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "user_terms",
                    scoreUserTerms(mode),
                    buildUserTermsJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "coaching_note",
                    scoreCoachingNote(mode, currentUserFit),
                    turnSpecificCoachingNote ?: integrity.userExperienceSummary
                )
            }

            PersonalizationProjectionMode.SOLVING_CONFRONTATION -> {
                addRankedField(
                    candidates,
                    "relationship_tone",
                    scoreRelationshipTone(mode),
                    toneBond.relationshipTone?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "speech_style",
                    scoreSpeechStyle(mode),
                    buildSpeechStyleJson(
                        replyRequest,
                        includePace = true,
                        includeRhythm = true,
                        includeVerbosity = true,
                        includeConfirmationStyle = false
                    )
                )

                addRankedField(
                    candidates,
                    "learning_bias",
                    scoreLearningBias(mode, currentUserFit),
                    buildLearningBiasJson(
                        replyRequest,
                        includePreferences = true,
                        includeProofPreference = true,
                        includeSetupPreference = false,
                        includeConfrontationPreference = true,
                        includeResolutionPreference = false
                    )
                )

                addRankedField(
                    candidates,
                    "technique_context",
                    scoreTechniqueContext(mode, currentTechnique, currentUserFit),
                    buildTechniqueContextJson(
                        replyRequest,
                        includeCurrentTechnique = true,
                        includeCurrentUserFit = true,
                        includeComfort = true,
                        includeRecentlyLearned = true,
                        includeChallenging = true
                    )
                )

                addRankedField(
                    candidates,
                    "avoid",
                    scoreAvoid(mode, currentUserFit),
                    buildAvoidJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "user_terms",
                    scoreUserTerms(mode),
                    buildUserTermsJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "coaching_note",
                    scoreCoachingNote(mode, currentUserFit),
                    turnSpecificCoachingNote ?: integrity.userExperienceSummary
                )
            }

            PersonalizationProjectionMode.SOLVING_RESOLUTION -> {
                addRankedField(
                    candidates,
                    "relationship_tone",
                    scoreRelationshipTone(mode),
                    toneBond.relationshipTone?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "speech_style",
                    scoreSpeechStyle(mode),
                    buildSpeechStyleJson(
                        replyRequest,
                        includePace = true,
                        includeRhythm = true,
                        includeVerbosity = true,
                        includeConfirmationStyle = false
                    )
                )

                addRankedField(
                    candidates,
                    "learning_bias",
                    scoreLearningBias(mode, currentUserFit),
                    buildLearningBiasJson(
                        replyRequest,
                        includePreferences = false,
                        includeProofPreference = false,
                        includeSetupPreference = false,
                        includeConfrontationPreference = false,
                        includeResolutionPreference = true
                    )
                )

                addRankedField(
                    candidates,
                    "technique_context",
                    scoreTechniqueContext(mode, currentTechnique, currentUserFit),
                    buildTechniqueContextJson(
                        replyRequest,
                        includeCurrentTechnique = true,
                        includeCurrentUserFit = true,
                        includeComfort = true,
                        includeRecentlyLearned = true,
                        includeChallenging = true
                    )
                )

                addRankedField(
                    candidates,
                    "avoid",
                    scoreAvoid(mode, currentUserFit),
                    buildAvoidJson(
                        replyRequest,
                        includeSpeechAvoid = true,
                        includeConfusionTriggers = false,
                        includeRecentFriction = true
                    )
                )

                addRankedField(
                    candidates,
                    "coaching_note",
                    scoreCoachingNote(mode, currentUserFit),
                    turnSpecificCoachingNote ?: integrity.userExperienceSummary
                )
            }

            PersonalizationProjectionMode.VALIDATION -> {
                addRankedField(
                    candidates,
                    "relationship_tone",
                    scoreRelationshipTone(mode),
                    toneBond.relationshipTone?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "speech_style",
                    scoreSpeechStyle(mode),
                    buildSpeechStyleJson(
                        replyRequest,
                        includePace = true,
                        includeRhythm = false,
                        includeVerbosity = true,
                        includeConfirmationStyle = true
                    )
                )

                addRankedField(
                    candidates,
                    "avoid",
                    scoreAvoid(mode, currentUserFit),
                    buildAvoidJson(
                        replyRequest,
                        includeSpeechAvoid = true,
                        includeConfusionTriggers = false,
                        includeRecentFriction = true
                    )
                )

                addRankedField(
                    candidates,
                    "user_terms",
                    scoreUserTerms(mode),
                    buildUserTermsJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "coaching_note",
                    scoreCoachingNote(mode, currentUserFit),
                    (turnSpecificCoachingNote ?: integrity.userExperienceSummary)?.let { truncateForMini(it, 140) }
                )
            }

            PersonalizationProjectionMode.SOCIAL -> {
                addRankedField(
                    candidates,
                    "relationship_tone",
                    scoreRelationshipTone(mode),
                    toneBond.relationshipTone?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "familiarity_preference",
                    94,
                    toneBond.familiarityPreference?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "humor_preference",
                    86,
                    toneBond.humorPreference?.name?.lowercase()
                )

                addRankedField(
                    candidates,
                    "name_usage_preference",
                    78,
                    toneBond.nameUsagePreference
                )

                addRankedField(
                    candidates,
                    "speech_style",
                    scoreSpeechStyle(mode),
                    buildSpeechStyleJson(replyRequest)
                )

                addRankedField(
                    candidates,
                    "user_terms",
                    scoreUserTerms(mode),
                    buildUserTermsJson(replyRequest)
                )

                val metaphorPolicy = learning.metaphorPolicy
                addRankedField(
                    candidates,
                    "metaphor_policy",
                    scoreMetaphorPolicy(mode, currentUserFit),
                    if (metaphorPolicy.allowed || metaphorPolicy.domains.isNotEmpty() || !metaphorPolicy.frequency.isNullOrBlank()) {
                        metaphorPolicy.toJson()
                    } else null
                )

                addRankedField(
                    candidates,
                    "coaching_note",
                    scoreCoachingNote(mode, currentUserFit),
                    turnSpecificCoachingNote ?: integrity.userExperienceSummary
                )
            }

            PersonalizationProjectionMode.MINIMAL -> {
                addRankedField(
                    candidates,
                    "speech_style",
                    scoreSpeechStyle(mode),
                    buildSpeechStyleJson(
                        replyRequest,
                        includePace = true,
                        includeRhythm = false,
                        includeVerbosity = true,
                        includeConfirmationStyle = false
                    )
                )

                addRankedField(
                    candidates,
                    "avoid",
                    scoreAvoid(mode, currentUserFit),
                    buildAvoidJson(
                        replyRequest,
                        includeSpeechAvoid = true,
                        includeConfusionTriggers = false,
                        includeRecentFriction = false
                    )
                )

                addRankedField(
                    candidates,
                    "user_terms",
                    scoreUserTerms(mode),
                    buildUserTermsJson(replyRequest)
                )
            }
        }

        projectRankedFields(out, mode, candidates)
        return out
    }

    fun projectCtaContext(replyRequest: ReplyRequestV1): JSONObject {
        val t = replyRequest.turn
        val cta = replyRequest.ctaContract
        val story = t.story
        val routeReturnAllowed = t.turnRouteReturnAllowed

        val stageEnum =
            when ((story?.stage ?: "").trim().uppercase()) {
                "SETUP" -> com.contextionary.sudoku.conductor.StoryStage.SETUP
                "CONFRONTATION" -> com.contextionary.sudoku.conductor.StoryStage.CONFRONTATION
                "RESOLUTION" -> com.contextionary.sudoku.conductor.StoryStage.RESOLUTION
                else -> null
            }

        return JSONObject().apply {
            put("pending_after", t.pendingAfter ?: JSONObject.NULL)

            if (story != null) {
                val storyCta = JSONObject().apply {
                    put("stage", story.stage ?: JSONObject.NULL)
                    put("step_id", story.stepId ?: JSONObject.NULL)
                    put("ready_for_commit", story.readyForCommit ?: JSONObject.NULL)
                    put("canonical_position_kind", story.canonicalPositionKind ?: JSONObject.NULL)
                    put("canonical_head_kind", story.canonicalHeadKind ?: JSONObject.NULL)
                    put("canonical_pending_kind", story.canonicalPendingKind ?: JSONObject.NULL)
                }
                put("story_cta", storyCta)
            }

            if (cta != null) {
                val ctaMini = JSONObject().apply {
                    put("family", cta.family.name)
                    put("owner_kind", cta.ownerKind)
                    put("route_moment", cta.routeMoment.name)
                    put("expected_response_type", cta.expectedResponseType.name)
                    put("ask_mode", cta.askMode.name)
                    put("closure_intent", cta.closureIntent ?: JSONObject.NULL)
                    put("bridge_intent", cta.bridgeIntent ?: JSONObject.NULL)
                    put("ask_intent", cta.askIntent ?: JSONObject.NULL)
                    put("allow_followup", cta.allowFollowUp)
                    put("allow_return_to_route", cta.allowReturnToRoute && routeReturnAllowed)
                    put("allow_route_mutation", cta.allowRouteMutation)
                    put("focus_cell_ref", cta.focusCellRef ?: JSONObject.NULL)
                    put("focus_house_ref", cta.focusHouseRef ?: JSONObject.NULL)
                    put("focus_digit", cta.focusDigit ?: JSONObject.NULL)
                    put("technique_name", cta.techniqueName ?: JSONObject.NULL)
                    put("tone_style", cta.toneStyle.name)
                    put("banned_phrases", JSONArray().apply { cta.bannedPhrases.forEach { put(it) } })
                    put("policy", cta.policy?.toJson() ?: JSONObject.NULL)
                }
                put("cta_contract", ctaMini)

                val ctaPolicyMini = JSONObject().apply {
                    put("family", cta.policy?.family?.name ?: cta.family.name)
                    put("route_moment", cta.routeMoment.name)
                    put(
                        "expected_response_type",
                        cta.policy?.expectedResponseType?.name ?: cta.expectedResponseType.name
                    )
                    put("ask_mode", cta.policy?.askMode?.name ?: cta.askMode.name)
                    put("must_offer_followup_choice", cta.policy?.mustOfferFollowUpChoice ?: false)
                    put("must_offer_return_choice", cta.policy?.mustOfferReturnChoice ?: false)
                    put("must_not_advance_stage", cta.policy?.mustNotAdvanceStage ?: false)
                    put("must_reference_focus_scope", cta.policy?.mustReferenceFocusScope ?: false)
                    put("allow_internal_jargon", cta.policy?.allowInternalJargon ?: false)
                }
                put("cta_policy", ctaPolicyMini)

                val surfaceRules = JSONArray()

                com.contextionary.sudoku.conductor.solving.SolvingPromptParts
                    .ctaSurfaceRulesForStage(stageEnum)
                    .forEach { surfaceRules.put(it) }

                if (
                    cta.family == CtaFamilyV1.USER_DETOUR_FOLLOWUP_OR_RETURN ||
                    cta.family == CtaFamilyV1.USER_LOCAL_REPAIR_CONFIRM
                ) {
                    com.contextionary.sudoku.conductor.solving.SolvingPromptParts
                        .detourReturnSurfaceRulesV1()
                        .forEach { surfaceRules.put(it) }
                }

                put("cta_surface_rules", surfaceRules)

                put(
                    "cta_question_shape",
                    com.contextionary.sudoku.conductor.solving.SolvingPromptParts
                        .ctaQuestionShapeForStage(stageEnum)
                )

                val preferredEnding =
                    when (cta.family) {
                        CtaFamilyV1.APP_SETUP_DISCOVERY ->
                            "Close the setup, focus the user on the local area, then ask one precise discovery question."

                        CtaFamilyV1.APP_CONFRONTATION_PROOF_STEP ->
                            "Close the proof point, bridge to the next local inference, then ask one precise proof-step question."

                        CtaFamilyV1.APP_RESOLUTION_COMMIT ->
                            "State the proved conclusion, bridge to action, then ask for placement or permission to apply."

                        CtaFamilyV1.APP_POST_COMMIT_CONTINUE ->
                            "Close the completed move, then offer next-step continuation."

                        CtaFamilyV1.USER_DETOUR_FOLLOWUP_OR_RETURN ->
                            "Close the detour answer, bridge back to the paused move, then offer a clear choice: return now or ask one more question."

                        CtaFamilyV1.USER_DETOUR_FOLLOWUP_ONLY ->
                            "State what is still unresolved, keep the detour open, then ask one narrow clarification question."

                        CtaFamilyV1.USER_ROUTE_CONTROL_CONFIRM ->
                            "Confirm the high-impact route change explicitly before any route mutation."

                        CtaFamilyV1.USER_LOCAL_REPAIR_CONFIRM ->
                            "Acknowledge the local discrepancy, then ask whether to fix this locally or broaden the repair."

                        else ->
                            "End with one clear user-facing CTA."
                    }

                put("cta_preferred_ending_shape", preferredEnding)
            }
        }
    }

    fun projectSolvabilityContext(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.SOLVABILITY_STATUS.name)
        return if (payload != null) {
            JSONObject().apply {
                put("type", FactBundleV1.Type.SOLVABILITY_STATUS.name)
                put("payload", payload)
            }
        } else {
            JSONObject()
        }
    }

    // ---------------------------------------------------------------------
    // Onboarding / confirming projectors
    // ---------------------------------------------------------------------

    fun projectOnboardingContext(replyRequest: ReplyRequestV1): JSONObject {
        val t = replyRequest.turn
        return JSONObject().apply {
            put("opening_turn", replyRequest.openingTurn)
            put("phase", t.phase)
            put("mode", t.mode)
            put("workflow_summary", replyRequest.decision.summary)
            put("assistant_style_header", replyRequest.assistantStyleHeader ?: "")
        }
    }









    fun projectConfirmingContext(replyRequest: ReplyRequestV1): JSONObject {
        val t = replyRequest.turn

        val confirmingPackets = collectFacts(
            replyRequest,
            setOf(
                FactBundleV1.Type.CONFIRMING_RETAKE_PACKET.name,
                FactBundleV1.Type.CONFIRMING_MISMATCH_PACKET.name,
                FactBundleV1.Type.CONFIRMING_CONFLICT_PACKET.name,
                FactBundleV1.Type.CONFIRMING_VISUAL_VERIFY_PACKET.name,
                FactBundleV1.Type.CONFIRMING_NOT_UNIQUE_PACKET.name,
                FactBundleV1.Type.CONFIRMING_FINALIZE_PACKET.name
            )
        )

        return JSONObject().apply {
            put("phase", t.phase)
            put("mode", t.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("pending_after", t.pendingAfter ?: JSONObject.NULL)
            put("solvability", projectSolvabilityContext(replyRequest))
            put("confirming_packets", confirmingPackets)
        }
    }

    private fun clarificationCellRefOrNullV1(
        rowHint: Any?,
        colHint: Any?
    ): String? {
        val row = (rowHint as? Number)?.toInt()
        val col = (colHint as? Number)?.toInt()
        return if (row != null && col != null && row in 1..9 && col in 1..9) {
            "r${row}c${col}"
        } else {
            null
        }
    }

    private fun buildPreferredClarificationQuestionV1(
        kind: String,
        prompt: String,
        rowHintPresent: Boolean,
        colHintPresent: Boolean,
        digitHintPresent: Boolean,
        cellRef: String?
    ): String {
        val cellSpoken =
            cellRef?.let {
                val r = it.substringAfter("r").substringBefore("c").toIntOrNull()
                val c = it.substringAfter("c").toIntOrNull()
                if (r != null && c != null) "row $r, column $c" else null
            }

        val rowMention =
            Regex("row\\s*(\\d+)", RegexOption.IGNORE_CASE)
                .find(prompt)
                ?.groupValues
                ?.getOrNull(1)

        val colMention =
            Regex("column\\s*(\\d+)", RegexOption.IGNORE_CASE)
                .find(prompt)
                ?.groupValues
                ?.getOrNull(1)

        val digitMention =
            Regex("digit\\s*(\\d+)", RegexOption.IGNORE_CASE)
                .find(prompt)
                ?.groupValues
                ?.getOrNull(1)

        return when {
            kind.equals("WORKFLOW", ignoreCase = true) &&
                    !rowHintPresent &&
                    !colHintPresent &&
                    !digitHintPresent &&
                    prompt.contains("row", ignoreCase = true) &&
                    prompt.contains("column", ignoreCase = true) &&
                    prompt.contains("box", ignoreCase = true) ->
                "Do you mean a row, a column, or a box?"

            rowHintPresent && !colHintPresent && digitHintPresent ->
                "I caught row ${rowMention ?: "that row"} and digit ${digitMention ?: "that digit"}. Which column do you mean?"

            !rowHintPresent && colHintPresent && digitHintPresent ->
                "I caught column ${colMention ?: "that column"} and digit ${digitMention ?: "that digit"}. Which row do you mean?"

            cellSpoken != null && !digitHintPresent ->
                "I caught $cellSpoken. Which digit do you want to check there?"

            !rowHintPresent && !colHintPresent && digitHintPresent ->
                "Which cell, row, column, or box do you want to check for digit ${digitMention ?: "that digit"}?"

            !rowHintPresent && !colHintPresent ->
                "Which row and column do you mean?"

            !rowHintPresent ->
                "Which row do you mean?"

            !colHintPresent ->
                "Which column do you mean?"

            !digitHintPresent ->
                "Which digit do you mean?"

            else ->
                prompt.ifBlank { "Could you clarify that?" }
        }
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    fun projectPendingContextChannel(replyRequest: ReplyRequestV1): JSONObject {
        val payload =
            findFactPayload(replyRequest, FactBundleV1.Type.PENDING_CONTEXT_V1.name) ?: return JSONObject()

        val pendingType = payload.optString("pending_type")
        val expectedAnswerKind = payload.optString("expected_answer_kind")
        val prompt = payload.optString("prompt")

        val rowHintPresent = !payload.isNull("row_hint")
        val colHintPresent = !payload.isNull("col_hint")
        val digitHintPresent = !payload.isNull("digit_hint")
        val cellRefPresent = !payload.isNull("cell_ref")

        val out = JSONObject().apply {
            put("pending_type", payload.opt("pending_type") ?: JSONObject.NULL)
            put("expected_answer_kind", payload.opt("expected_answer_kind") ?: JSONObject.NULL)
            put("prompt", payload.opt("prompt") ?: JSONObject.NULL)
            put("row_hint", payload.opt("row_hint") ?: JSONObject.NULL)
            put("col_hint", payload.opt("col_hint") ?: JSONObject.NULL)
            put("digit_hint", payload.opt("digit_hint") ?: JSONObject.NULL)
            put("cell_ref", payload.opt("cell_ref") ?: JSONObject.NULL)
        }

        if (pendingType == "ask_clarification") {
            val kind = payload.optString("kind")
            val effectiveCellRef =
                (payload.opt("cell_ref") as? String)?.takeIf { it.isNotBlank() }
                    ?: clarificationCellRefOrNullV1(
                        rowHint = payload.opt("row_hint"),
                        colHint = payload.opt("col_hint")
                    )

            val missingFields = JSONArray()
            var clarificationSubtype = "generic_missing_target"
            var clarificationGoal = "obtain the missing detail needed to proceed"

            if (
                kind.equals("WORKFLOW", ignoreCase = true) &&
                !rowHintPresent &&
                !colHintPresent &&
                !digitHintPresent &&
                prompt.contains("row", ignoreCase = true) &&
                prompt.contains("column", ignoreCase = true) &&
                prompt.contains("box", ignoreCase = true)
            ) {
                clarificationSubtype = "region_missing"
                clarificationGoal = "identify whether the user means a row, column, or box"
                missingFields.put("region")
            } else {
                if (!rowHintPresent && !colHintPresent && !cellRefPresent) {
                    missingFields.put("target")
                }
                if (!digitHintPresent && expectedAnswerKind.contains("digit", ignoreCase = true)) {
                    missingFields.put("digit")
                }
                if (!rowHintPresent && !colHintPresent && prompt.contains("row", ignoreCase = true).not()) {
                    // Leave generic target missing rather than falsely forcing row/col.
                } else {
                    if (!rowHintPresent) missingFields.put("row")
                    if (!colHintPresent) missingFields.put("col")
                }
            }

            val preferredQuestion =
                buildPreferredClarificationQuestionV1(
                    kind = kind,
                    prompt = prompt,
                    rowHintPresent = rowHintPresent,
                    colHintPresent = colHintPresent,
                    digitHintPresent = digitHintPresent,
                    cellRef = effectiveCellRef
                )

            val clarificationIsActionable =
                preferredQuestion.isNotBlank() &&
                        (missingFields.length() > 0 || effectiveCellRef != null || digitHintPresent)

            out.put("contract_family", "pending_clarification")
            out.put("clarification_subtype", clarificationSubtype)
            out.put("missing_fields", missingFields)
            out.put("clarification_goal", clarificationGoal)
            out.put("preferred_user_facing_question", preferredQuestion)
            out.put("clarification_is_actionable", clarificationIsActionable)
            out.put("response_shape", "one_short_user_facing_question")
            out.put("fallback_response_policy", "ask_bounded_question_not_meta_status")
            if (effectiveCellRef != null) {
                out.put("cell_ref", effectiveCellRef)
            }
            if (kind.isNotBlank()) {
                out.put("kind", kind)
            }
            return out
        }

        val pendingTypeLc = pendingType.lowercase()
        val expectedAnswerKindLc = expectedAnswerKind.lowercase()
        val promptLc = prompt.lowercase()

        var contractFamily = "pending_gate"
        var contractSubtype = "generic_pending_gate"
        var targetScopeKind = "unknown"
        var targetKind = "unknown"
        var responseShape = "generic"

        val isRegionScoped =
            promptLc.contains("row") || promptLc.contains("column") || promptLc.contains("box")
        val isCellScoped =
            cellRefPresent || (rowHintPresent && colHintPresent)
        val expectsDigit =
            digitHintPresent || expectedAnswerKindLc.contains("digit") || promptLc.contains("digit")
        val asksAsIs =
            promptLc.contains("as is") ||
                    promptLc.contains("already correct") ||
                    promptLc.contains("leave") ||
                    promptLc.contains("fine")
        val asksInterpretation =
            promptLc.contains("interpret") ||
                    promptLc.contains("read this") ||
                    promptLc.contains("is this what you see") ||
                    promptLc.contains("does this look right")

        if (isRegionScoped) {
            targetScopeKind = "region"
            targetKind = "row_col_or_box"
        } else if (isCellScoped) {
            targetScopeKind = "cell"
            targetKind = "single_cell"
        }

        when {
            asksInterpretation -> {
                contractFamily = "pending_transactional"
                contractSubtype = "pending_interpretation_confirm"
                responseShape = "confirm_interpretation"
            }

            isRegionScoped && asksAsIs -> {
                contractFamily = "pending_transactional"
                contractSubtype = "pending_region_confirm_as_is"
                responseShape = "confirm_region_as_is"
            }

            isRegionScoped && expectsDigit -> {
                contractFamily = "pending_transactional"
                contractSubtype = "pending_region_confirm_to_digits"
                responseShape = "confirm_region_to_digits"
            }

            isCellScoped && asksAsIs -> {
                contractFamily = "pending_transactional"
                contractSubtype = "pending_cell_confirm_as_is"
                responseShape = "confirm_cell_as_is"
            }

            isCellScoped && expectsDigit -> {
                contractFamily = "pending_transactional"
                contractSubtype = "pending_cell_confirm_to_digit"
                responseShape = "confirm_cell_to_digit"
            }

            expectsDigit -> {
                contractFamily = "pending_transactional"
                contractSubtype = "pending_digit_provide"
                responseShape = "provide_digit"
            }
        }

        out.put("contract_family", contractFamily)
        out.put("contract_subtype", contractSubtype)
        out.put("target_scope_kind", targetScopeKind)
        out.put("target_kind", targetKind)
        out.put("response_shape", responseShape)

        val missingFields = JSONArray()
        if (contractSubtype == "pending_cell_confirm_as_is" || contractSubtype == "pending_cell_confirm_to_digit") {
            if (!cellRefPresent && !rowHintPresent) missingFields.put("row")
            if (!cellRefPresent && !colHintPresent) missingFields.put("col")
        }
        if (contractSubtype == "pending_region_confirm_as_is" || contractSubtype == "pending_region_confirm_to_digits") {
            if (!isRegionScoped) missingFields.put("region")
        }
        if ((contractSubtype == "pending_cell_confirm_to_digit" ||
                    contractSubtype == "pending_region_confirm_to_digits" ||
                    contractSubtype == "pending_digit_provide") && !digitHintPresent) {
            missingFields.put("digit")
        }
        out.put("missing_fields", missingFields)

        if (contractSubtype.startsWith("pending_")) {
            out.put("preferred_user_facing_question", prompt.ifBlank { JSONObject.NULL })
        }

        return out
    }

    fun projectGridValidationContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.STRUCTURAL_VALIDITY.name,
            FactBundleV1.Type.CONFLICT_SET.name,
            FactBundleV1.Type.DUPLICATES_BY_HOUSE.name,
            FactBundleV1.Type.UNRESOLVED_SET.name,
            FactBundleV1.Type.MISMATCH_SET.name,
            FactBundleV1.Type.MISMATCH_EXPLANATION.name,
            FactBundleV1.Type.CONFLICT_EXPLANATION.name,
            FactBundleV1.Type.FOCUS_CELL_SNAPSHOT.name,
            FactBundleV1.Type.RETAKE_RECOMMENDATION.name,
            FactBundleV1.Type.OCR_CONFIDENCE_CELL.name,
            FactBundleV1.Type.OCR_CONFIDENCE_SUMMARY.name,
            FactBundleV1.Type.SEAL_STATUS.name,
            FactBundleV1.Type.RECENT_MUTATION_RESULT.name,
            FactBundleV1.Type.CONFIRMING_RETAKE_PACKET.name,
            FactBundleV1.Type.CONFIRMING_MISMATCH_PACKET.name,
            FactBundleV1.Type.CONFIRMING_CONFLICT_PACKET.name,
            FactBundleV1.Type.CONFIRMING_VISUAL_VERIFY_PACKET.name,
            FactBundleV1.Type.CONFIRMING_NOT_UNIQUE_PACKET.name,
            FactBundleV1.Type.CONFIRMING_FINALIZE_PACKET.name
        )

        val facts = collectFacts(replyRequest, included)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)

            val solvability = projectSolvabilityContext(replyRequest)
            if (solvability.length() > 0) {
                put("solvability", solvability)
            }
        }
    }

    fun projectGridCandidateContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.CANDIDATE_STATE_CELL.name,
            FactBundleV1.Type.CELLS_WITH_N_CANDS_SET.name,
            FactBundleV1.Type.BIVALUE_CELLS_SET.name,
            FactBundleV1.Type.HOUSE_CANDIDATE_MAP.name,
            FactBundleV1.Type.DIGIT_CANDIDATE_FREQUENCY.name,
            FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1.name,
            FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1.name
        )

        val facts = collectFacts(replyRequest, included)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)
        }
    }

    fun projectGridOcrTrustContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.OCR_CONFIDENCE_CELL.name,
            FactBundleV1.Type.OCR_CONFIDENCE_SUMMARY.name,
            FactBundleV1.Type.FOCUS_CELL_SNAPSHOT.name,
            FactBundleV1.Type.MISMATCH_EXPLANATION.name,
            FactBundleV1.Type.CONFLICT_EXPLANATION.name
        )

        val facts = collectFacts(replyRequest, included)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)

            val solvability = projectSolvabilityContext(replyRequest)
            if (solvability.length() > 0) {
                put("solvability", solvability)
            }
        }
    }

    fun projectGridContentsContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.GRID_SNAPSHOT.name,
            FactBundleV1.Type.CELL_STATUS_BUNDLE.name,
            FactBundleV1.Type.HOUSE_STATUS_BUNDLE.name,
            FactBundleV1.Type.HOUSES_COMPLETION_RANKING.name,
            FactBundleV1.Type.DIGIT_LOCATIONS_BUNDLE.name
        )

        val facts = collectFacts(replyRequest, included)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)
        }
    }

    fun projectGridChangelogContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.RECENT_MUTATION_RESULT.name
        )

        val facts = collectFacts(replyRequest, included)
        val latestMutation =
            findFactPayload(replyRequest, FactBundleV1.Type.RECENT_MUTATION_RESULT.name)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)

            if (latestMutation != null) {
                put(
                    "latest_mutation_summary",
                    JSONObject().apply {
                        put("kind", latestMutation.opt("kind") ?: JSONObject.NULL)
                        put("status", latestMutation.opt("status") ?: JSONObject.NULL)
                        put("cell_ref", latestMutation.opt("cell_ref") ?: JSONObject.NULL)
                        put("scope", latestMutation.opt("scope") ?: JSONObject.NULL)
                        put("digit", latestMutation.opt("digit") ?: JSONObject.NULL)
                        put("message", latestMutation.opt("message") ?: JSONObject.NULL)
                    }
                )
            }
        }
    }

    fun projectGridMutationContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.RECENT_MUTATION_RESULT.name,
            FactBundleV1.Type.STRUCTURAL_VALIDITY.name,
            FactBundleV1.Type.CONFLICT_SET.name,
            FactBundleV1.Type.DUPLICATES_BY_HOUSE.name,
            FactBundleV1.Type.UNRESOLVED_SET.name,
            FactBundleV1.Type.MISMATCH_SET.name,
            FactBundleV1.Type.SEAL_STATUS.name
        )

        val facts = collectFacts(replyRequest, included)
        val latestMutation =
            findFactPayload(replyRequest, FactBundleV1.Type.RECENT_MUTATION_RESULT.name)
        val pendingPayload =
            findFactPayload(replyRequest, FactBundleV1.Type.PENDING_CONTEXT_V1.name)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)

            if (latestMutation != null) {
                put(
                    "mutation_intent",
                    JSONObject().apply {
                        put("kind", latestMutation.opt("kind") ?: JSONObject.NULL)
                        put("status", latestMutation.opt("status") ?: JSONObject.NULL)
                        put("cell_ref", latestMutation.opt("cell_ref") ?: JSONObject.NULL)
                        put("scope", latestMutation.opt("scope") ?: JSONObject.NULL)
                        put("digit", latestMutation.opt("digit") ?: JSONObject.NULL)
                    }
                )
            }

            if (pendingPayload != null) {
                put(
                    "pending_support",
                    JSONObject().apply {
                        put("pending_type", pendingPayload.opt("pending_type") ?: JSONObject.NULL)
                        put("prompt", pendingPayload.opt("prompt") ?: JSONObject.NULL)
                        put("cell_ref", pendingPayload.opt("cell_ref") ?: JSONObject.NULL)
                        put("row_hint", pendingPayload.opt("row_hint") ?: JSONObject.NULL)
                        put("col_hint", pendingPayload.opt("col_hint") ?: JSONObject.NULL)
                        put("digit_hint", pendingPayload.opt("digit_hint") ?: JSONObject.NULL)
                    }
                )
            }

            val changelog = projectGridChangelogContext(replyRequest)
            if (changelog.length() > 0) {
                put("changelog", changelog)
            }
        }
    }

    fun projectSolvingSupportContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1.name,
            FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1.name,
            FactBundleV1.Type.HANDOVER_NOTE_V1.name,
            FactBundleV1.Type.STORY_CONTEXT_V1.name,
            FactBundleV1.Type.CTA_PACKET_V1.name,
            FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name,
            FactBundleV1.Type.SETUP_REPLY_PACKET_V1.name,
            FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1.name,
            FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1.name
        )

        val facts = collectFacts(replyRequest, included)
        val stepClarification =
            findFactPayload(replyRequest, FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1.name)
        val returnToRoute =
            findFactPayload(replyRequest, FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1.name)
        val solvingStep =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)
        val handover =
            findFactPayload(replyRequest, FactBundleV1.Type.HANDOVER_NOTE_V1.name)
        val storyContext =
            findFactPayload(replyRequest, FactBundleV1.Type.STORY_CONTEXT_V1.name)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)

            put(
                "support_kind",
                when {
                    stepClarification != null -> "step_clarification"
                    returnToRoute != null -> "return_to_route"
                    else -> "solving_support"
                }
            )

            if (solvingStep != null) {
                put(
                    "step_summary",
                    JSONObject().apply {
                        put("step_id", solvingStep.opt("step_id") ?: JSONObject.NULL)
                        put("technique", solvingStep.opt("technique") ?: JSONObject.NULL)
                        put("target_cell", solvingStep.opt("target_cell") ?: JSONObject.NULL)
                        put("target_digit", solvingStep.opt("target_digit") ?: JSONObject.NULL)
                    }
                )
            }

            if (stepClarification != null) {
                put(
                    "step_clarification_summary",
                    JSONObject().apply {
                        put("question_kind", stepClarification.opt("question_kind") ?: JSONObject.NULL)
                        put("target_cell", stepClarification.opt("target_cell") ?: JSONObject.NULL)
                        put("digit", stepClarification.opt("digit") ?: JSONObject.NULL)
                        put("house_ref", stepClarification.opt("house_ref") ?: JSONObject.NULL)
                        put("prompt", stepClarification.opt("prompt") ?: JSONObject.NULL)
                    }
                )
            }

            if (returnToRoute != null) {
                put(
                    "return_to_route_summary",
                    JSONObject().apply {
                        put("route_kind", returnToRoute.opt("route_kind") ?: JSONObject.NULL)
                        put("resume_point", returnToRoute.opt("resume_point") ?: JSONObject.NULL)
                        put("prompt", returnToRoute.opt("prompt") ?: JSONObject.NULL)
                    }
                )
            }

            if (handover != null) {
                put("handover_note", handover)
            }

            if (storyContext != null) {
                put("story_context", storyContext)
            }
        }
    }

    fun projectDetourContext(replyRequest: ReplyRequestV1): JSONObject {
        val included = setOf(
            FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1.name,
            FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1.name,
            FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1.name,
            FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1.name,
            FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1.name,
            FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1.name,
            FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1.name,
            FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name,
            FactBundleV1.Type.CTA_PACKET_V1.name,
            FactBundleV1.Type.HANDOVER_NOTE_V1.name
        )

        val facts = collectFacts(replyRequest, included)
        val proofChallenge =
            findFactPayload(replyRequest, FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1.name)
        val targetQuery =
            findFactPayload(replyRequest, FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1.name)
        val neighborQuery =
            findFactPayload(replyRequest, FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1.name)
        val reasoningCheck =
            findFactPayload(replyRequest, FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1.name)
        val alternativeTechnique =
            findFactPayload(replyRequest, FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
        val localMoveSearch =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1.name)
        val routeComparison =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1.name)
        val returnToRoute =
            findFactPayload(replyRequest, FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1.name)
        val solvingStep =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("decision_summary", replyRequest.decision.summary)
            put("facts", facts)

            put(
                "detour_kind",
                when {
                    proofChallenge != null -> "proof_challenge"
                    targetQuery != null -> "target_cell_query"
                    neighborQuery != null -> "neighbor_cell_query"
                    reasoningCheck != null -> "reasoning_check"
                    alternativeTechnique != null -> "alternative_technique"
                    localMoveSearch != null -> "local_move_search"
                    routeComparison != null -> "route_comparison"
                    else -> "generic_detour"
                }
            )

            if (solvingStep != null) {
                put(
                    "step_summary",
                    JSONObject().apply {
                        put("step_id", solvingStep.opt("step_id") ?: JSONObject.NULL)
                        put("technique", solvingStep.opt("technique") ?: JSONObject.NULL)
                        put("target_cell", solvingStep.opt("target_cell") ?: JSONObject.NULL)
                        put("target_digit", solvingStep.opt("target_digit") ?: JSONObject.NULL)
                    }
                )
            }

            if (proofChallenge != null) put("proof_challenge", proofChallenge)
            if (targetQuery != null) put("target_cell_query", targetQuery)
            if (neighborQuery != null) put("neighbor_cell_query", neighborQuery)
            if (reasoningCheck != null) put("reasoning_check", reasoningCheck)
            if (alternativeTechnique != null) put("alternative_technique", alternativeTechnique)
            if (localMoveSearch != null) put("local_move_search", localMoveSearch)
            if (routeComparison != null) put("route_comparison", routeComparison)

            if (returnToRoute != null) {
                put(
                    "return_to_route_summary",
                    JSONObject().apply {
                        put("route_kind", returnToRoute.opt("route_kind") ?: JSONObject.NULL)
                        put("resume_point", returnToRoute.opt("resume_point") ?: JSONObject.NULL)
                        put("prompt", returnToRoute.opt("prompt") ?: JSONObject.NULL)
                    }
                )
            }
        }
    }

    fun projectPreferenceContext(replyRequest: ReplyRequestV1): JSONObject {
        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("user_text", replyRequest.turn.userText)
            put("decision_summary", replyRequest.decision.summary)
            put("assistant_style_header", replyRequest.assistantStyleHeader ?: "")
            put("recent_turns", takeFirstNRecentTurns(replyRequest.recentTurns, 4))
            put("user_tally", replyRequest.userTally?.toJson() ?: JSONObject.NULL)
            put("assistant_tally", replyRequest.assistantTally?.toJson() ?: JSONObject.NULL)
        }
    }

    fun projectMetaContext(replyRequest: ReplyRequestV1): JSONObject {
        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("user_text", replyRequest.turn.userText)
            put("decision_kind", replyRequest.decision.decisionKind.name)
            put("decision_summary", replyRequest.decision.summary)
            put("recent_turns", takeFirstNRecentTurns(replyRequest.recentTurns, 4))
            put("story", projectStoryHeaderMini(replyRequest.turn.story))
            put("pending_after", replyRequest.turn.pendingAfter ?: JSONObject.NULL)
            put("focus_after", replyRequest.turn.focusAfter ?: JSONObject.NULL)
        }
    }

    fun projectHelpContext(replyRequest: ReplyRequestV1): JSONObject {
        val glossary = projectGlossaryMini(replyRequest)
        val techniqueCard = projectTechniqueCardMini(replyRequest)

        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("user_text", replyRequest.turn.userText)
            put("decision_summary", replyRequest.decision.summary)
            put("recent_turns", takeFirstNRecentTurns(replyRequest.recentTurns, 3))

            if (glossary.length() > 0) {
                put("glossary_mini", glossary)
            }
            if (techniqueCard.length() > 0) {
                put("technique_card_mini", techniqueCard)
            }
        }
    }

    fun projectFreeTalkContext(replyRequest: ReplyRequestV1): JSONObject {
        return JSONObject().apply {
            put("phase", replyRequest.turn.phase)
            put("mode", replyRequest.turn.mode)
            put("user_text", replyRequest.turn.userText)
            put("assistant_style_header", replyRequest.assistantStyleHeader ?: "")
            put("recent_turns", takeFirstNRecentTurns(replyRequest.recentTurns, 4))
            put("user_tally", replyRequest.userTally?.toJson() ?: JSONObject.NULL)
            put("assistant_tally", replyRequest.assistantTally?.toJson() ?: JSONObject.NULL)
        }
    }

    // ---------------------------------------------------------------------
    // Story projectors
    // ---------------------------------------------------------------------

    fun projectSetupStorySlice(replyRequest: ReplyRequestV1): JSONObject {
        return projectStoryHeaderMini(replyRequest.turn.story)
    }

    fun projectConfrontationStorySlice(replyRequest: ReplyRequestV1): JSONObject {
        return projectStoryHeaderMini(replyRequest.turn.story)
    }

    fun projectResolutionStorySlice(replyRequest: ReplyRequestV1): JSONObject {
        return projectStoryHeaderMini(replyRequest.turn.story)
    }

    // ---------------------------------------------------------------------
    // Fact-bundle projectors
    // ---------------------------------------------------------------------

    fun projectSetupStepSlice(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)
            ?: return JSONObject()

        val out = JSONObject()
        copyIfPresent(payload, out, "step_id")
        copyIfPresent(payload, out, "technique")
        copyIfPresent(payload, out, "label")
        copyIfPresent(payload, out, "focus_cell")
        copyIfPresent(payload, out, "focus_digit")
        copyIfPresent(payload, out, "target_cell")
        copyIfPresent(payload, out, "target_digit")
        copyIfPresent(payload, out, "spoiler_policy")
        copyIfPresent(payload, out, "why_now")
        copyIfPresent(payload, out, "headline")
        copyIfPresent(payload, out, "setup")
        copyIfPresent(payload, out, "pattern")
        copyIfPresent(payload, out, "subset")
        copyIfPresent(payload, out, "intersection")
        copyIfPresent(payload, out, "houses")
        copyIfPresent(payload, out, "cta")
        return out
    }

    fun projectConfrontationStepSlice(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)
            ?: return JSONObject()

        val out = JSONObject()
        copyIfPresent(payload, out, "step_id")
        copyIfPresent(payload, out, "technique")
        copyIfPresent(payload, out, "focus_cell")
        copyIfPresent(payload, out, "focus_digit")
        copyIfPresent(payload, out, "target_cell")
        copyIfPresent(payload, out, "target_digit")
        copyIfPresent(payload, out, "proof")
        copyIfPresent(payload, out, "proof_payload")
        copyIfPresent(payload, out, "pattern_member_proofs")
        copyIfPresent(payload, out, "witnesses")
        copyIfPresent(payload, out, "eliminations")
        copyIfPresent(payload, out, "houses")
        copyIfPresent(payload, out, "cta")
        return out
    }

    fun projectResolutionStepSlice(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)
            ?: return JSONObject()

        val out = JSONObject()
        copyIfPresent(payload, out, "step_id")
        copyIfPresent(payload, out, "technique")
        copyIfPresent(payload, out, "target_cell")
        copyIfPresent(payload, out, "target_digit")
        copyIfPresent(payload, out, "commit")
        copyIfPresent(payload, out, "resolution")
        copyIfPresent(payload, out, "final_reason")
        copyIfPresent(payload, out, "cta")
        return out
    }



    fun projectSetupReplyPacket(replyRequest: ReplyRequestV1): JSONObject {
        val payload =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.SETUP_REPLY_PACKET_V1)
                ?: return JSONObject()

        val technique = payload.optJSONObject("technique") ?: JSONObject()
        val focus = payload.optJSONObject("focus") ?: JSONObject()
        val patternStructure = payload.optJSONObject("pattern_structure") ?: JSONObject()
        val triggerOverview = payload.optJSONObject("trigger_overview") ?: JSONObject()
        val bridge = payload.optJSONObject("bridge") ?: JSONObject()
        val setupOnlyLine = payload.optJSONObject("setup_only_line") ?: JSONObject()
        val cta = payload.optJSONObject("cta") ?: JSONObject()
        val support = payload.optJSONObject("support") ?: JSONObject()
        val boundedTriggerRows = payload.optJSONArray("bounded_trigger_rows") ?: JSONArray()

        val doctrine = payload.optString("setup_doctrine")
        val isPatternFirst = doctrine.equals("PATTERN_FIRST", ignoreCase = true)
        val isLensFirst = doctrine.equals("LENS_FIRST", ignoreCase = true)

        val setupProfileWire = payload.optString("setup_profile")
        val archetypeWire = payload.optString("archetype")
        val isIntersectionSetup =
            setupProfileWire.equals("INTERSECTIONS_SETUP", ignoreCase = true) ||
                    archetypeWire.equals("INTERSECTIONS", ignoreCase = true) ||
                    support.optBoolean("intersection_family", false)

        fun houseLabel(house: JSONObject?): String {
            val type = house?.optString("type").orEmpty()
            val index = house?.optInt("index1to9", -1) ?: -1
            return when {
                type == "row" && index in 1..9 -> "row $index"
                type == "col" && index in 1..9 -> "column $index"
                type == "box" && index in 1..9 -> "box $index"
                else -> "the house"
            }
        }

        val out = JSONObject()
        out.put("schema_version", "setup_reply_packet_projection_v3")
        out.put("setup_profile", payload.opt("setup_profile") ?: JSONObject.NULL)
        out.put("setup_doctrine", payload.opt("setup_doctrine") ?: JSONObject.NULL)
        out.put("archetype", payload.opt("archetype") ?: JSONObject.NULL)

        out.put("technique", JSONObject().apply {
            put(
                "name",
                technique.opt("app_name")
                    ?: technique.opt("spoken_technique_name")
                    ?: technique.opt("name")
                    ?: technique.opt("technique_name")
                    ?: JSONObject.NULL
            )
            put(
                "real_name",
                technique.opt("app_name")
                    ?: technique.opt("spoken_technique_name")
                    ?: technique.opt("real_name")
                    ?: technique.opt("name")
                    ?: technique.opt("technique_name")
                    ?: JSONObject.NULL
            )
            put("app_name", technique.opt("app_name") ?: JSONObject.NULL)
            put("engine_name", technique.opt("engine_name") ?: JSONObject.NULL)
            put("family", technique.opt("family") ?: JSONObject.NULL)
            put("difficulty_level", technique.opt("difficulty_level") ?: JSONObject.NULL)
            put(
                "one_line_personality",
                payload.opt("technique_entrance_label")
                    ?: technique.opt("app_name")
                    ?: technique.opt("spoken_technique_name")
                    ?: technique.opt("short_definition_summary")
                    ?: JSONObject.NULL
            )
        })

        if (isIntersectionSetup) {
            val triggerPattern = triggerOverview.optJSONObject("pattern") ?: JSONObject()
            val triggerExplanation = triggerOverview.optJSONObject("explanation_payload") ?: JSONObject()
            val bridgePayload = bridge.optJSONObject("payload") ?: JSONObject()

            val sourceHouse =
                patternStructure.optJSONObject("source_house")
                    ?: triggerPattern.optJSONObject("source_house")
                    ?: JSONObject()

            val crossHouse =
                patternStructure.optJSONObject("cross_house")
                    ?: triggerPattern.optJSONObject("cross_house")
                    ?: JSONObject()

            val boxHouse =
                patternStructure.optJSONObject("box_house")
                    ?: triggerPattern.optJSONObject("box_house")
                    ?: JSONObject()

            val lineHouse =
                patternStructure.optJSONObject("line_house")
                    ?: triggerPattern.optJSONObject("line_house")
                    ?: JSONObject()

            val overlapCells =
                patternStructure.optJSONArray("overlap_cells")
                    ?: triggerPattern.optJSONArray("overlap_cells")
                    ?: JSONArray()

            val patternCells =
                patternStructure.optJSONArray("pattern_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: overlapCells

            val sourceOutsideCells =
                patternStructure.optJSONArray("source_outside_overlap_cells")
                    ?: triggerPattern.optJSONArray("source_outside_overlap_cells")
                    ?: JSONArray()

            val crossOutsideCells =
                patternStructure.optJSONArray("cross_outside_overlap_cells")
                    ?: triggerPattern.optJSONArray("cross_outside_overlap_cells")
                    ?: JSONArray()

            val forbiddenCrossCells =
                patternStructure.optJSONArray("forbidden_cross_cells")
                    ?: triggerPattern.optJSONArray("forbidden_cross_cells")
                    ?: bridgePayload.optJSONArray("forbidden_elsewhere_cells")
                    ?: JSONArray()

            val nativeSourceAuditRows =
                triggerExplanation.optJSONArray("setup_preferred_audit_rows")
                    ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_outside_open_seat_audit")
                        ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_witness_closure_rows")
                        ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_outside_overlap_setup_audit")
                        ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_setup_closure_walk")
                        ?.takeIf { it.length() > 0 }
                    ?: triggerExplanation.optJSONArray("source_house_outside_overlap_audit")
                        ?.takeIf { it.length() > 0 }
                    ?: boundedTriggerRows

            val overlapSurvivorCells =
                triggerExplanation.optJSONArray("overlap_survivor_cells")
                    ?.takeIf { it.length() > 0 }
                    ?: intersectionRowFallbackSurvivorsFromOverlap(overlapCells)

            val repeatedDigits =
                patternStructure.optJSONArray("repeated_candidate_digits")
                    ?.takeIf { it.length() > 0 }
                    ?: JSONArray().apply {
                        focus.opt("digit")?.takeIf { it != JSONObject.NULL }?.let { put(it) }
                    }

            val patternSubtype =
                patternStructure.optString("pattern_subtype")
                    .takeIf { it.isNotBlank() }
                    ?: triggerPattern.optString("pattern_subtype")
                        .takeIf { it.isNotBlank() }

            val patternCardinality =
                when {
                    patternCells.length() > 0 -> patternCells.length()
                    overlapSurvivorCells.length() > 0 -> overlapSurvivorCells.length()
                    else -> overlapCells.length()
                }

            val cardinalityName =
                when (patternCardinality) {
                    2 -> "pair"
                    3 -> "triple"
                    else -> "group"
                }

            val spokenTechniqueName =
                when {
                    !patternSubtype.isNullOrBlank() -> patternSubtype.replace('_', ' ')
                    else -> JSONObject.NULL
                }

            val subtypeExplanationLine =
                when {
                    patternCardinality in 2..3 ->
                        "The technique takes its name from how many overlap cells are holding the digit here: $patternCardinality cells makes this a $cardinalityName."
                    else ->
                        JSONObject.NULL
                }

            val sourceHouseLabel =
                if (sourceHouse.length() > 0) houseLabel(sourceHouse) else null

            val crossHouseLabel =
                if (crossHouse.length() > 0) houseLabel(crossHouse) else null

            val boxHouseLabel =
                if (boxHouse.length() > 0) houseLabel(boxHouse) else null

            val lineHouseLabel =
                if (lineHouse.length() > 0) houseLabel(lineHouse) else null

            val pressureDirection =
                when (patternStructure.optString("direction_mode").lowercase()) {
                    "claiming" -> "source_house_claims_digit_inside_box"
                    "pointing" -> "box_claims_digit_inside_line"
                    else -> "overlap_confines_digit_across_two_houses"
                }

            val controlOriginHouse =
                when (patternStructure.optString("direction_mode").lowercase()) {
                    "claiming" -> if (sourceHouse.length() > 0) sourceHouse else JSONObject.NULL
                    "pointing" -> if (boxHouse.length() > 0) boxHouse else JSONObject.NULL
                    else -> if (sourceHouse.length() > 0) sourceHouse else JSONObject.NULL
                }

            val controlReceiverHouse =
                when (patternStructure.optString("direction_mode").lowercase()) {
                    "claiming" -> if (boxHouse.length() > 0) boxHouse else JSONObject.NULL
                    "pointing" -> if (lineHouse.length() > 0) lineHouse else JSONObject.NULL
                    else -> if (crossHouse.length() > 0) crossHouse else JSONObject.NULL
                }

            val controlOriginHouseLabel =
                when {
                    controlOriginHouse is JSONObject && controlOriginHouse.length() > 0 -> houseLabel(controlOriginHouse)
                    else -> null
                }

            val controlReceiverHouseLabel =
                when {
                    controlReceiverHouse is JSONObject && controlReceiverHouse.length() > 0 -> houseLabel(controlReceiverHouse)
                    else -> null
                }

            val battlefieldHouse =
                when {
                    crossHouse.length() > 0 -> crossHouse
                    else -> JSONObject()
                }

            val battlefieldHouseLabel =
                if (battlefieldHouse.length() > 0) houseLabel(battlefieldHouse) else null

            val sourceConfinementLine =
                triggerExplanation.opt("forced_inward_reason")
                    ?: triggerExplanation.opt("forced_into_overlap_summary")
                    ?: payload.opt("pattern_birth_summary_line")
                    ?: payload.opt("pattern_birth_summary")
                    ?: JSONObject.NULL

            val territorialClaimLine =
                when {
                    controlOriginHouseLabel != null && controlReceiverHouseLabel != null ->
                        "$controlOriginHouseLabel is now telling $controlReceiverHouseLabel: my digit lives in our shared strip somewhere, so you cannot keep it anywhere else."
                    else ->
                        bridgePayload.opt("cross_house_permission_change")
                            ?: bridge.opt("summary")
                            ?: JSONObject.NULL
                }

            val battlefieldTeaserLine =
                when {
                    battlefieldHouseLabel != null ->
                        "The pattern is ready, and the next place to inspect is $battlefieldHouseLabel."
                    else ->
                        JSONObject.NULL
                }

            val patternRevealLine =
                payload.opt("pattern_completion_moment")
                    ?: payload.opt("pattern_birth_summary")
                    ?: triggerExplanation.opt("pattern_reveal_moment")
                    ?: JSONObject.NULL

            val structuralSignificance =
                bridge.opt("why_this_matters")
                    ?: bridge.opt("summary")
                    ?: payload.opt("why_this_technique_now")
                    ?: bridgePayload.opt("cross_house_permission_change")
                    ?: JSONObject.NULL

            val cardinalityRuleLine =
                when (patternStructure.optString("direction_mode").lowercase()) {
                    "claiming" ->
                        "When a digit is trapped in the overlap between a line and a box, the technique takes its name from how many overlap cells are holding that digit there."
                    "pointing" ->
                        "When a digit is trapped in the overlap between a box and a line, the technique takes its name from how many overlap cells are holding that digit there."
                    else ->
                        "When a digit is trapped in the overlap between two houses, the technique takes its name from how many overlap cells are holding that digit there."
                }

            val thisCaseSubtypeLine =
                when {
                    patternStructure.optString("direction_mode").equals("claiming", ignoreCase = true) && patternCardinality == 2 ->
                        "The technique takes its name from how many overlap cells are holding the digit. If two cells hold it, it is a claiming pair. If three hold it, it is a claiming triple. Here the digit is confined to exactly two overlap cells, so this is a claiming pair."
                    patternStructure.optString("direction_mode").equals("claiming", ignoreCase = true) && patternCardinality == 3 ->
                        "The technique takes its name from how many overlap cells are holding the digit. If two cells hold it, it is a claiming pair. If three hold it, it is a claiming triple. Here the digit is confined to exactly three overlap cells, so this is a claiming triple."
                    patternStructure.optString("direction_mode").equals("pointing", ignoreCase = true) && patternCardinality == 2 ->
                        "The technique takes its name from how many overlap cells are holding the digit. If two cells hold it, it is a pointing pair. If three hold it, it is a pointing triple. Here the digit is confined to exactly two overlap cells, so this is a pointing pair."
                    patternStructure.optString("direction_mode").equals("pointing", ignoreCase = true) && patternCardinality == 3 ->
                        "The technique takes its name from how many overlap cells are holding the digit. If two cells hold it, it is a pointing pair. If three hold it, it is a pointing triple. Here the digit is confined to exactly three overlap cells, so this is a pointing triple."
                    else ->
                        subtypeExplanationLine
                }

            fun projectIntersectionAuditWalk(): JSONArray {
                val filteredRows = ArrayList<JSONObject>()

                for (i in 0 until nativeSourceAuditRows.length()) {
                    val row = nativeSourceAuditRows.optJSONObject(i) ?: continue
                    val seatState = row.optString("seat_state")
                    val narratableInSetup =
                        if (row.has("narratable_in_setup")) row.optBoolean("narratable_in_setup", true) else true
                    val closureKind = row.optString("closure_kind")
                    val reasonKind = row.optString("reason_kind")

                    val keep =
                        (seatState.isBlank() || seatState.equals("open", ignoreCase = true)) &&
                                narratableInSetup &&
                                (
                                        closureKind.equals("witness_blocked", ignoreCase = true) ||
                                                closureKind.equals("digit_absent", ignoreCase = true) ||
                                                reasonKind.equals("single_cell_witness", ignoreCase = true) ||
                                                reasonKind.equals("candidate_absent", ignoreCase = true)
                                        )

                    if (keep) {
                        filteredRows.add(JSONObject(row.toString()))
                    }
                }

                val spokenRows = filteredRows
                    .take(3)

                return JSONArray().apply {
                    spokenRows.forEachIndexed { idx, row ->
                        put(
                            JSONObject().apply {
                                put("row_index", idx + 1)
                                put("audited_cell", row.opt("cell") ?: JSONObject.NULL)
                                put("audited_cell_label", row.opt("cell_label") ?: JSONObject.NULL)
                                put("digit", row.opt("digit") ?: JSONObject.NULL)
                                put("seat_state", row.opt("seat_state") ?: JSONObject.NULL)
                                put("digit_live_here", row.opt("digit_live_here") ?: JSONObject.NULL)
                                put("closure_kind", row.opt("closure_kind") ?: JSONObject.NULL)
                                put("reason_kind", row.opt("reason_kind") ?: JSONObject.NULL)
                                put("relation", row.opt("relation") ?: JSONObject.NULL)
                                put("witness_cell", row.opt("witness_cell") ?: JSONObject.NULL)
                                put("witness_cell_label", row.opt("witness_cell_label") ?: JSONObject.NULL)
                                put("witness_house", row.opt("house") ?: JSONObject.NULL)
                                put("witness_house_label", row.opt("house_label") ?: JSONObject.NULL)
                                put("spoken_reason", row.opt("spoken_reason") ?: JSONObject.NULL)
                                put("setup_spoken_line", row.opt("setup_spoken_line") ?: row.opt("spoken_reason") ?: JSONObject.NULL)
                                put("setup_priority", row.opt("setup_priority") ?: JSONObject.NULL)
                                put("narration_role", row.opt("narration_role") ?: JSONObject.NULL)
                            }
                        )
                    }
                }
            }

            fun projectIntersectionAuditSummary(): JSONObject {
                val witnessRows = JSONArray()
                val noncandidateRows = JSONArray()

                for (i in 0 until nativeSourceAuditRows.length()) {
                    val row = nativeSourceAuditRows.optJSONObject(i) ?: continue
                    val seatState = row.optString("seat_state")
                    val narratableInSetup =
                        if (row.has("narratable_in_setup")) row.optBoolean("narratable_in_setup", true) else true
                    if (!(seatState.isBlank() || seatState.equals("open", ignoreCase = true)) || !narratableInSetup) {
                        continue
                    }

                    when {
                        row.optString("closure_kind").equals("witness_blocked", ignoreCase = true) ||
                                row.optString("reason_kind").equals("single_cell_witness", ignoreCase = true) -> {
                            witnessRows.put(JSONObject(row.toString()))
                        }
                        row.optString("closure_kind").equals("digit_absent", ignoreCase = true) ||
                                row.optString("reason_kind").equals("candidate_absent", ignoreCase = true) -> {
                            noncandidateRows.put(JSONObject(row.toString()))
                        }
                    }
                }

                return JSONObject().apply {
                    put("witness_rows", witnessRows)
                    put("open_noncandidate_rows", noncandidateRows)
                    put("witness_count", witnessRows.length())
                    put("open_noncandidate_count", noncandidateRows.length())
                }
            }

            out.put("basic_setup_surface", JSONObject().apply {
                put("focus_house", if (sourceHouse.length() > 0) sourceHouse else focus.opt("house") ?: JSONObject.NULL)
                put("focus_digit", focus.opt("digit") ?: JSONObject.NULL)
                put("lens_question", JSONObject.NULL)
                put(
                    "why_this_technique_fits_now",
                    payload.opt("why_this_technique_now")
                        ?: bridgePayload.opt("cross_house_permission_change")
                        ?: JSONObject.NULL
                )
            })

            out.put("advanced_setup_surface", JSONObject().apply {
                put("local_pattern_zone", if (boxHouse.length() > 0) boxHouse else patternStructure.opt("zone_house") ?: JSONObject.NULL)
                put("ordered_pattern_members", JSONArray())
                put("repeated_digits", repeatedDigits)
                put("shared_surviving_digits", repeatedDigits)
                put("pattern_symmetry_or_shape", patternRevealLine)
                put("why_the_pattern_matters", structuralSignificance)

                put("earned_reveal_surface", JSONObject().apply {
                    put("must_audit_before_naming", true)
                    put("must_force_question_inward", true)
                    put("must_name_subtype_after_survivors", true)
                    put("source_house_label", sourceHouseLabel ?: JSONObject.NULL)
                    put("cross_house_label", crossHouseLabel ?: JSONObject.NULL)
                    put("overlap_survivor_cells", overlapSurvivorCells)
                    put("outside_audit_walk", projectIntersectionAuditWalk())
                })

                put("crossroads_stage", JSONObject().apply {
                    put("intersection_label", patternStructure.opt("intersection_label") ?: "box-line crossroads")
                    put("source_house", if (sourceHouse.length() > 0) sourceHouse else JSONObject.NULL)
                    put("cross_house", if (crossHouse.length() > 0) crossHouse else JSONObject.NULL)
                    put("box_house", if (boxHouse.length() > 0) boxHouse else JSONObject.NULL)
                    put("line_house", if (lineHouse.length() > 0) lineHouse else JSONObject.NULL)
                    put("source_house_label", sourceHouseLabel ?: JSONObject.NULL)
                    put("cross_house_label", crossHouseLabel ?: JSONObject.NULL)
                    put("box_house_label", boxHouseLabel ?: JSONObject.NULL)
                    put("line_house_label", lineHouseLabel ?: JSONObject.NULL)
                    put("overlap_cells", overlapCells)
                    put(
                        "opening_line",
                        triggerExplanation.opt("forced_inward_reason")
                            ?: triggerExplanation.opt("forced_into_overlap_summary")
                            ?: payload.opt("pattern_birth_summary_line")
                            ?: payload.opt("pattern_birth_summary")
                            ?: JSONObject.NULL
                    )
                    put("visual_opening_focus", "crossroads_overlap")
                })

                put("source_confinement_stage", JSONObject().apply {
                    put("source_house", if (sourceHouse.length() > 0) sourceHouse else JSONObject.NULL)
                    put("source_house_label", sourceHouseLabel ?: JSONObject.NULL)
                    put("focus_digit", focus.opt("digit") ?: JSONObject.NULL)
                    put("source_outside_overlap_cells", sourceOutsideCells)
                    put("outside_open_seat_cells", triggerExplanation.opt("outside_open_seat_cells") ?: sourceOutsideCells)
                    put("outside_audit_walk", projectIntersectionAuditWalk())
                    put("outside_open_seat_summary", projectIntersectionAuditSummary())
                    put("forced_inward_summary", sourceConfinementLine)
                    put(
                        "forced_inward_reason",
                        triggerExplanation.opt("forced_inward_reason")
                            ?: sourceConfinementLine
                    )
                    put("overlap_survivor_cells", overlapSurvivorCells)
                    put("survivor_count", overlapSurvivorCells.length())
                })

                put("outside_audit_walk", projectIntersectionAuditWalk())

                put("territorial_collapse_moment", JSONObject().apply {
                    put(
                        "source_house_has_run_out_of_room",
                        sourceConfinementLine
                    )
                    put(
                        "digit_forced_into_overlap",
                        sourceConfinementLine
                    )
                    put("source_outside_overlap_cells", sourceOutsideCells)
                    put("outside_audit_walk", projectIntersectionAuditWalk())
                    put("overlap_survivor_cells", overlapSurvivorCells)
                    put("survivor_count", overlapSurvivorCells.length())
                })


                put("pattern_reveal", JSONObject().apply {
                    put("pattern_subtype", patternSubtype ?: JSONObject.NULL)
                    put("direction_mode", patternStructure.opt("direction_mode") ?: JSONObject.NULL)
                    put("cardinality", patternCardinality)
                    put("cardinality_name", cardinalityName)
                    put("spoken_technique_name", spokenTechniqueName)
                    put("subtype_explanation_line", subtypeExplanationLine)
                    put("cardinality_rule_line", cardinalityRuleLine)
                    put("this_case_subtype_line", thisCaseSubtypeLine ?: JSONObject.NULL)
                    put("pattern_cells", patternCells)
                    put("overlap_survivor_cells", overlapSurvivorCells)
                    put("shared_surviving_digits", repeatedDigits)
                    put("pattern_completion_moment", patternRevealLine)
                    put(
                        "pattern_name_line",
                        triggerExplanation.opt("pattern_reveal_moment")
                            ?: patternRevealLine
                            ?: JSONObject.NULL
                    )
                    put(
                        "pattern_origin_story",
                        triggerExplanation.opt("forced_inward_reason")
                            ?: sourceConfinementLine
                    )
                })



                put("structural_significance", JSONObject().apply {
                    put("cross_house_now_restricted", bridgePayload.opt("cross_house_now_restricted") ?: JSONObject.NULL)
                    put("forbidden_elsewhere_cells_preview", forbiddenCrossCells)
                    put("pressure_direction", pressureDirection)
                    put("control_origin_house", controlOriginHouse)
                    put("control_receiver_house", controlReceiverHouse)
                    put("control_origin_house_label", controlOriginHouseLabel ?: JSONObject.NULL)
                    put("control_receiver_house_label", controlReceiverHouseLabel ?: JSONObject.NULL)
                    put(
                        "territorial_control_line",
                        territorialClaimLine
                    )
                    put("why_this_pattern_matters_now", structuralSignificance)
                    put("must_stop_before_target_effect", true)
                })

                put("zone_stage", JSONObject().apply {
                    put("local_pattern_zone", if (boxHouse.length() > 0) boxHouse else patternStructure.opt("zone_house") ?: JSONObject.NULL)
                    put("opening_stage_kind", "crossroads_overlap")
                })

                put("member_walk", JSONArray())
            })

            out.put("setup_doctrine_surface", JSONObject().apply {
                put("opening_energy", "crossroads_tension")
                put(
                    "visual_staging_hint",
                    "Open in the source house and the hunted digit, close the outside seats step by step, force the question inward into the overlap, reveal the surviving overlap cells, then name the subtype and the cross-house pressure before any downstream target effect is spent."
                )
                put(
                    "pattern_entrance_line",
                    sourceConfinementLine
                )
                put("must_name_both_houses", true)
                put("must_end_on_control_not_solution", true)
                put("explicit_source_house_audit_required", true)
                put("native_setup_spine", "source_house_audit_then_overlap_survivors")
                put("must_earn_the_pattern_name", true)
                put("must_not_open_with_finished_pattern_summary", true)
                put("must_not_skip_to_overlap_without_source_closure_walk", true)
                put("must_prefer_explicit_blocker_lines_over_generic_closure_summary", true)
                put("must_keep_handoff_natural_not_formulaic", true)
                put(
                    "north_star_rhythm_hint",
                    "Open with curiosity, close the outside seats, force the question inward, reveal the overlap survivors, then name the subtype and the cross-house pressure."
                )
                put("source_house_label", sourceHouseLabel ?: JSONObject.NULL)
                put("cross_house_label", crossHouseLabel ?: JSONObject.NULL)
                put("battlefield_house_label", battlefieldHouseLabel ?: JSONObject.NULL)
                put("battlefield_teaser_line", battlefieldTeaserLine)
            })

            out.put("stage_bridge", JSONObject().apply {
                put("summary", JSONObject.NULL)
                put(
                    "target_effect_summary",
                    bridge.opt("summary")
                        ?: bridgePayload.opt("cross_house_permission_change")
                        ?: JSONObject.NULL
                )
                put("setup_only_line", setupOnlyLine.opt("summary") ?: JSONObject.NULL)
                put("spoiler_safe", true)
                put("must_stop_before_target_effect", true)
                put("battlefield_house", if (battlefieldHouse.length() > 0) battlefieldHouse else JSONObject.NULL)
                put("battlefield_house_label", battlefieldHouseLabel ?: JSONObject.NULL)
                put("battlefield_teaser_line", battlefieldTeaserLine)
            })

            out.put("cta", JSONObject().apply {
                put("kind", cta.opt("kind") ?: JSONObject.NULL)

                val naturalHandoffEnabled =
                    battlefieldTeaserLine != JSONObject.NULL &&
                            battlefieldTeaserLine != null &&
                            battlefieldHouseLabel != null

                put(
                    "style",
                    if (naturalHandoffEnabled) "natural_setup_handoff" else JSONObject.NULL
                )

                put("battlefield_house_label", battlefieldHouseLabel ?: JSONObject.NULL)

                put(
                    "preferred_question_shape",
                    if (naturalHandoffEnabled) {
                        "Would you like to follow that pressure into the next part of the move?"
                    } else {
                        JSONObject.NULL
                    }
                )
            })

            out.put("support", JSONObject().apply {
                put("doctrine_is_lens_first", false)
                put("doctrine_is_pattern_first", true)
                put("has_focus_house", sourceHouse.length() > 0 || focus.has("house"))
                put("has_focus_digit", focus.has("digit"))
                put("has_lens_question", false)
                put("has_ordered_pattern_members", false)
                put("has_repeated_candidate_digits", repeatedDigits.length() > 0)
                put("has_member_walk", false)
                put("has_structural_significance", structuralSignificance != JSONObject.NULL && structuralSignificance != null)
                put("has_why_this_technique_now", payload.has("why_this_technique_now"))
                put("has_bridge_summary", bridge.has("summary") || bridgePayload.length() > 0)
                put("has_cta_kind", cta.has("kind"))
                put("setup_target_spend_forbidden", true)
                put("speech_surface_mode", "NORTH_STAR_INTERSECTION_SETUP_STAGE_OWNER")

                put("intersection_family", true)
                put("source_house_label", sourceHouseLabel ?: JSONObject.NULL)
                put("cross_house_label", crossHouseLabel ?: JSONObject.NULL)
                put("battlefield_house_label", battlefieldHouseLabel ?: JSONObject.NULL)
                put("overlap_cell_count", overlapCells.length())
                put("source_outside_overlap_count", sourceOutsideCells.length())
                put("cross_outside_overlap_count", crossOutsideCells.length())
                put("forbidden_cross_cell_count", forbiddenCrossCells.length())
                put("outside_audit_row_count", nativeSourceAuditRows.length())
                put("has_explicit_source_outside_audit", nativeSourceAuditRows.length() > 0)
                put("has_source_confinement_stage", true)
                put("has_overlap_cells", overlapCells.length() > 0)
                put("has_overlap_survivors_after_audit", overlapSurvivorCells.length() > 0)

                put("has_subtype_explanation_line", subtypeExplanationLine != JSONObject.NULL && subtypeExplanationLine != null)
                put("has_cardinality_rule_line", cardinalityRuleLine.isNotBlank())
                put("has_this_case_subtype_line", thisCaseSubtypeLine != JSONObject.NULL && thisCaseSubtypeLine != null)
                put("has_house_to_house_pressure_line", territorialClaimLine != JSONObject.NULL && territorialClaimLine != null)
                put("has_battlefield_teaser", battlefieldTeaserLine != JSONObject.NULL && battlefieldTeaserLine != null)
                put(
                    "has_natural_setup_handoff",
                    battlefieldTeaserLine != JSONObject.NULL &&
                            battlefieldTeaserLine != null &&
                            battlefieldHouseLabel != null
                )

                put("has_forbidden_cross_cells_preview", forbiddenCrossCells.length() > 0)


                put("setup_law_explicit_audit_required", true)
            })

            return out
        }

        val orderedPatternMembers = patternStructure.optJSONArray("ordered_members") ?: JSONArray()

        val sharedSurvivingDigits =
            patternStructure.optJSONArray("repeated_candidate_digits")
                ?.takeIf { it.length() > 0 }
                ?: orderedPatternMembers
                    .optJSONObject(0)
                    ?.optJSONArray("remaining_candidate_digits")
                ?: JSONArray()

        val localPatternZone =
            patternStructure.opt("zone_house")
                ?: focus.opt("house")
                ?: JSONObject.NULL

        val patternRevealLine =
            payload.opt("pattern_completion_moment")
                ?: payload.opt("pattern_birth_summary")
                ?: JSONObject.NULL

        val structuralSignificance =
            when {
                isPatternFirst ->
                    payload.opt("why_this_technique_now") ?: JSONObject.NULL

                else ->
                    payload.opt("why_this_technique_now")
                        ?: bridge.opt("summary")
                        ?: JSONObject.NULL
            }

        fun projectMemberWalk(): JSONArray = JSONArray().apply {
            for (i in 0 until orderedPatternMembers.length()) {
                val member = orderedPatternMembers.optJSONObject(i) ?: continue
                val boundedRow = boundedTriggerRows.optJSONObject(i)

                put(
                    JSONObject().apply {
                        put("member_index", i + 1)
                        put("member_cell", member.opt("member_cell") ?: JSONObject.NULL)
                        put(
                            "surviving_digits",
                            member.optJSONArray("remaining_candidate_digits") ?: JSONArray()
                        )
                        put(
                            "shared_surviving_digits",
                            sharedSurvivingDigits
                        )
                        put(
                            "claimed_digits",
                            member.optJSONArray("claimed_candidate_digits") ?: JSONArray()
                        )
                        put(
                            "witness_rows",
                            member.optJSONArray("witness_rows") ?: JSONArray()
                        )
                        put(
                            "grouped_witness_summary",
                            boundedRow?.opt("grouped_witness_summary")
                                ?: member.opt("grouped_witness_summary")
                                ?: JSONObject.NULL
                        )
                    }
                )
            }
        }

        out.put("basic_setup_surface", JSONObject().apply {
            put("focus_house", focus.opt("house") ?: JSONObject.NULL)
            put("focus_digit", focus.opt("digit") ?: JSONObject.NULL)
            put(
                "lens_question",
                focus.opt("lens_question")
                    ?: payload.opt("lens_question")
                    ?: JSONObject.NULL
            )
            put(
                "why_this_technique_fits_now",
                payload.opt("why_this_technique_now") ?: JSONObject.NULL
            )
        })

        out.put("advanced_setup_surface", JSONObject().apply {
            put("local_pattern_zone", localPatternZone)
            put("ordered_pattern_members", orderedPatternMembers)
            put("repeated_digits", sharedSurvivingDigits)
            put("shared_surviving_digits", sharedSurvivingDigits)
            put("pattern_symmetry_or_shape", patternRevealLine)
            put("why_the_pattern_matters", structuralSignificance)

            put("zone_stage", JSONObject().apply {
                put("local_pattern_zone", localPatternZone)
                put(
                    "opening_stage_kind",
                    if (isPatternFirst) "local_pattern_zone" else JSONObject.NULL
                )
            })

            put("member_walk", projectMemberWalk())

            put("pattern_reveal", JSONObject().apply {
                put("shared_surviving_digits", sharedSurvivingDigits)
                put("pattern_completion_moment", patternRevealLine)
                put(
                    "completion_member_cell",
                    patternStructure.opt("completion_member_cell") ?: JSONObject.NULL
                )
                put(
                    "symmetry_line",
                    payload.opt("pattern_birth_summary")
                        ?: payload.opt("pattern_completion_moment")
                        ?: JSONObject.NULL
                )
            })

            put("structural_significance", JSONObject().apply {
                put("why_this_pattern_matters_now", structuralSignificance)
                put(
                    "must_stop_before_target_effect",
                    isPatternFirst
                )
            })
        })

        out.put("setup_doctrine_surface", JSONObject().apply {
            put(
                "opening_energy",
                when {
                    isPatternFirst -> "pattern_emergence"
                    isLensFirst -> "quiet_lens"
                    else -> JSONObject.NULL
                }
            )
            put(
                "visual_staging_hint",
                when {
                    isPatternFirst ->
                        "Let the local pattern come into view before spending it; make the shared surviving digits and member symmetry perceptible before naming the downstream effect."
                    isLensFirst ->
                        "Let the house/digit lens feel like a quiet spotlight rather than a proof list."
                    else -> JSONObject.NULL
                }
            )
            put(
                "pattern_entrance_line",
                if (isPatternFirst) {
                    payload.opt("pattern_birth_summary")
                        ?: payload.opt("pattern_completion_moment")
                        ?: JSONObject.NULL
                } else {
                    JSONObject.NULL
                }
            )
        })

        out.put("stage_bridge", JSONObject().apply {
            put(
                "summary",
                if (isPatternFirst) JSONObject.NULL else bridge.opt("summary") ?: JSONObject.NULL
            )
            put(
                "target_effect_summary",
                if (isPatternFirst) bridge.opt("summary") ?: JSONObject.NULL else JSONObject.NULL
            )
            put("setup_only_line", setupOnlyLine.opt("summary") ?: JSONObject.NULL)
            put("spoiler_safe", true)
            put("must_stop_before_target_effect", isPatternFirst)
        })

        out.put("cta", JSONObject().apply {
            put("kind", cta.opt("kind") ?: JSONObject.NULL)
        })

        out.put("support", JSONObject().apply {
            put("doctrine_is_lens_first", isLensFirst)
            put("doctrine_is_pattern_first", isPatternFirst)
            put("has_focus_house", focus.has("house"))
            put("has_focus_digit", focus.has("digit"))
            put("has_lens_question", focus.has("lens_question") || payload.has("lens_question"))
            put("has_ordered_pattern_members", orderedPatternMembers.length() > 0)
            put("has_repeated_candidate_digits", sharedSurvivingDigits.length() > 0)
            put("has_member_walk", orderedPatternMembers.length() > 0)
            put("has_structural_significance", structuralSignificance != JSONObject.NULL && structuralSignificance != null)
            put("has_why_this_technique_now", payload.has("why_this_technique_now"))
            put("has_bridge_summary", bridge.has("summary"))
            put("has_cta_kind", cta.has("kind"))
            put("setup_target_spend_forbidden", isPatternFirst)
            put("speech_surface_mode", "NORTH_STAR_SETUP_STAGE_OWNER")
        })

        return out
    }

    private fun intersectionRowFallbackSurvivorsFromOverlap(overlapCells: JSONArray): JSONArray =
        JSONArray().apply {
            for (i in 0 until overlapCells.length()) {
                put(overlapCells.opt(i))
            }
        }




    fun projectConfrontationReplyPacket(replyRequest: ReplyRequestV1): JSONObject {
        val payload =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1)
                ?: return JSONObject()

        val technique = payload.optJSONObject("technique") ?: JSONObject()
        val target = payload.optJSONObject("target") ?: JSONObject()
        val collapse = payload.optJSONObject("collapse") ?: JSONObject()
        val preCommitLine = payload.optJSONObject("pre_commit_line") ?: JSONObject()
        val cta = payload.optJSONObject("cta") ?: JSONObject()
        val support = payload.optJSONObject("support") ?: JSONObject()
        val triggerReference = payload.optJSONObject("trigger_reference") ?: JSONObject()
        val triggerEffect = payload.optJSONObject("trigger_effect") ?: JSONObject()

        val targetResolutionTruth = payload.optJSONObject("target_resolution_truth") ?: JSONObject()
        val targetTruthSupport = targetResolutionTruth.optJSONObject("support") ?: JSONObject()

        val proofRows = payload.optJSONArray("target_proof_rows") ?: JSONArray()
        val techniqueBlockerRows =
            targetTruthSupport.optJSONArray("technique_blocker_rows") ?: JSONArray()
        val peerBlockerRows =
            targetTruthSupport.optJSONArray("peer_blocker_rows") ?: JSONArray()

        val doctrine = payload.optString("confrontation_doctrine")
        val isPatternFirst = doctrine.equals("PATTERN_FIRST", ignoreCase = true)
        val isLensFirst = doctrine.equals("LENS_FIRST", ignoreCase = true)

        fun projectLeanRows(rows: JSONArray, limit: Int): JSONArray =
            JSONArray().apply {
                val max = minOf(rows.length(), limit)
                for (i in 0 until max) {
                    val raw = rows.opt(i)
                    when (raw) {
                        is JSONObject -> {
                            put(
                                JSONObject().apply {
                                    put("cell", raw.opt("cell") ?: raw.opt("claimed_cell") ?: JSONObject.NULL)
                                    put("digit", raw.opt("digit") ?: JSONObject.NULL)
                                    put("house_scope", raw.opt("house_scope") ?: JSONObject.NULL)
                                    put("because", raw.opt("because") ?: JSONObject.NULL)
                                    put("spoken_line", raw.opt("spoken_line") ?: JSONObject.NULL)
                                }
                            )
                        }

                        is String -> {
                            put(
                                JSONObject().apply {
                                    put("cell", JSONObject.NULL)
                                    put("digit", JSONObject.NULL)
                                    put("house_scope", JSONObject.NULL)
                                    put("because", JSONObject.NULL)
                                    put("spoken_line", raw)
                                }
                            )
                        }
                    }
                }
            }

        fun ladderStep(
            stepKind: String,
            actor: String,
            summary: Any?,
            rows: JSONArray = JSONArray(),
            required: Boolean = false
        ): JSONObject =
            JSONObject().apply {
                put("step_kind", stepKind)
                put("actor", actor)
                put("summary", summary ?: JSONObject.NULL)
                put("rows", rows)
                put("required", required)
            }

        val projectedProofRows = projectLeanRows(proofRows, 6)
        val projectedPeerRows = projectLeanRows(peerBlockerRows, if (isPatternFirst) 6 else 4)
        val projectedTechniqueRows = projectLeanRows(techniqueBlockerRows, 4)

        val targetSpotlightLine =
            support.opt("target_spotlight_line")
                ?: targetResolutionTruth.opt("summary")
                ?: JSONObject.NULL

        val twoActorHonestyLine =
            collapse.opt("two_layer_honesty_line")
                ?: support.opt("actor_structure")
                ?: JSONObject.NULL

        val survivorRevealLine =
            support.opt("survivor_reveal_line")
                ?: collapse.opt("summary")
                ?: JSONObject.NULL

        val heroEntranceLine =
            when {
                projectedTechniqueRows.length() > 0 ->
                    triggerReference.opt("summary")
                        ?: triggerEffect.opt("summary")
                        ?: "Now the named technique steps in to close the last meaningful exits."

                else ->
                    JSONObject.NULL
            }

        fun projectPerformedTwoActorLadder(): JSONArray = JSONArray().apply {
            put(
                ladderStep(
                    stepKind = "target_spotlight",
                    actor = "spotlight",
                    summary = targetSpotlightLine,
                    required = true
                )
            )

            if (projectedPeerRows.length() > 0) {
                put(
                    ladderStep(
                        stepKind = "ordinary_witnesses",
                        actor = "ordinary_witnesses",
                        summary = "Let the ordinary witnesses thin the crowd first.",
                        rows = projectedPeerRows,
                        required = support.optBoolean("ordinary_witness_first_required", false) || isPatternFirst
                    )
                )
            }

            if (projectedTechniqueRows.length() > 0) {
                put(
                    ladderStep(
                        stepKind = "hero_technique",
                        actor = "hero_technique",
                        summary = heroEntranceLine,
                        rows = projectedTechniqueRows,
                        required = support.optBoolean("technique_finishing_cut_required", false) || isPatternFirst
                    )
                )
            }

            put(
                ladderStep(
                    stepKind = "survivor_reveal",
                    actor = "survivor",
                    summary = survivorRevealLine,
                    required = true
                )
            )

            if (twoActorHonestyLine != JSONObject.NULL) {
                put(
                    ladderStep(
                        stepKind = "two_actor_honesty",
                        actor = "narrator",
                        summary = twoActorHonestyLine,
                        required = isPatternFirst
                    )
                )
            }

            put(
                ladderStep(
                    stepKind = "proof_complete_boundary",
                    actor = "narrator",
                    summary = collapse.opt("summary") ?: JSONObject.NULL,
                    required = true
                )
            )
        }

        val performedTwoActorLadder = projectPerformedTwoActorLadder()

        val out = JSONObject()
        out.put("schema_version", "confrontation_reply_packet_projection_v3")
        out.put("proof_profile", payload.opt("proof_profile") ?: JSONObject.NULL)
        out.put("confrontation_doctrine", payload.opt("confrontation_doctrine") ?: JSONObject.NULL)
        out.put("archetype", payload.opt("archetype") ?: JSONObject.NULL)

        out.put("technique", JSONObject().apply {
            put(
                "name",
                technique.opt("name")
                    ?: technique.opt("technique_name")
                    ?: JSONObject.NULL
            )
            put(
                "real_name",
                technique.opt("real_name")
                    ?: technique.opt("name")
                    ?: technique.opt("technique_name")
                    ?: JSONObject.NULL
            )
            put("family", technique.opt("family") ?: JSONObject.NULL)
        })

        out.put("target_frame", JSONObject().apply {
            val primaryHouse = target.opt("primary_house") ?: JSONObject.NULL
            val targetCell = target.opt("cell") ?: JSONObject.NULL
            val targetDigit = target.opt("target_digit") ?: JSONObject.NULL

            put("target_cell", targetCell)
            put("primary_house", primaryHouse)
            put("target_digit", targetDigit)
            put("can_say_target_digit", target.optBoolean("can_say_target_digit", false))
            put(
                "battlefield_kind",
                if (targetResolutionTruth.optString("elimination_kind") == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT") {
                    "HOUSE_BATTLEFIELD"
                } else {
                    "CELL_BATTLEFIELD"
                }
            )
            put("live_question_or_spotlight", targetSpotlightLine)
        })

        out.put("basic_confrontation_surface", JSONObject().apply {
            put("ordered_rival_seat_eliminations", projectedProofRows)
            put("final_surviving_seat", JSONObject().apply {
                put("cell", target.opt("cell") ?: JSONObject.NULL)
                put("digit", collapse.opt("surviving_digit") ?: target.opt("target_digit") ?: JSONObject.NULL)
                put("summary", survivorRevealLine)
            })
        })

        out.put("advanced_confrontation_surface", JSONObject().apply {
            put("ordinary_witness_elimination_group", projectedPeerRows)
            put("technique_elimination_group", projectedTechniqueRows)
            put("performed_two_actor_ladder", performedTwoActorLadder)
            put("hero_entrance_line", heroEntranceLine)
            put("two_actor_honesty_line", twoActorHonestyLine)
            put("two_actor_story_summary", twoActorHonestyLine)
            put("survivor_digit", collapse.opt("surviving_digit") ?: JSONObject.NULL)
            put("survivor_reveal_line", survivorRevealLine)
        })

        out.put("confrontation_doctrine_surface", JSONObject().apply {
            put(
                "performance_mode",
                when {
                    isPatternFirst -> "two_actor_duet"
                    isLensFirst -> "live_narrowing"
                    else -> JSONObject.NULL
                }
            )
            put("target_spotlight_line", targetSpotlightLine)
            put(
                "ordinary_witness_pressure_hint",
                if (projectedPeerRows.length() > 0) {
                    "Let the ordinary witnesses thin the crowd before the hero technique acts."
                } else {
                    JSONObject.NULL
                }
            )
            put(
                "technique_entrance_hint",
                if (projectedTechniqueRows.length() > 0) {
                    "Give the named technique a real entrance beat: it closes the last meaningful exits after the crowd has already thinned."
                } else {
                    JSONObject.NULL
                }
            )
            put("hero_entrance_line", heroEntranceLine)
            put("two_actor_honesty_line", twoActorHonestyLine)
            put("survivor_reveal_hint", survivorRevealLine)
        })

        out.put("proof_complete_boundary", JSONObject().apply {
            put("proof_complete_line", collapse.opt("summary") ?: JSONObject.NULL)
            put("pre_commit_line", preCommitLine.opt("summary") ?: JSONObject.NULL)
            put("two_actor_honesty_line", twoActorHonestyLine)
            put("must_stop_before_commit", true)
        })

        out.put("cta", JSONObject().apply {
            put("kind", cta.opt("kind") ?: JSONObject.NULL)
        })

        out.put("support", JSONObject().apply {
            put("has_basic_ordered_eliminations", projectedProofRows.length() > 0)
            put("has_ordinary_witness_group", projectedPeerRows.length() > 0)
            put("has_technique_group", projectedTechniqueRows.length() > 0)
            put("has_performed_two_actor_ladder", performedTwoActorLadder.length() > 0)
            put(
                "has_two_actor_story_summary",
                collapse.has("two_layer_honesty_line") || support.has("actor_structure")
            )
            put("has_hero_entrance_line", heroEntranceLine != JSONObject.NULL)
            put("has_pre_commit_line", preCommitLine.has("summary"))
            put("has_cta_kind", cta.has("kind"))
            put(
                "ordinary_witness_first_required",
                support.optBoolean("ordinary_witness_first_required", false)
            )
            put(
                "technique_finishing_cut_required",
                support.optBoolean("technique_finishing_cut_required", false)
            )
            put("speech_surface_mode", "NORTH_STAR_CONFRONTATION_STAGE_OWNER")
        })

        return out
    }

    fun projectResolutionReplyPacket(replyRequest: ReplyRequestV1): JSONObject {
        val payload =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1)
                ?: return JSONObject()

        val technique = payload.optJSONObject("technique") ?: JSONObject()
        val commit = payload.optJSONObject("commit") ?: JSONObject()
        val recap = payload.optJSONObject("recap") ?: JSONObject()
        val techniqueContribution = payload.optJSONObject("technique_contribution") ?: JSONObject()


        val finalForcing = payload.optJSONObject("final_forcing") ?: JSONObject()
        val honesty = payload.optJSONObject("honesty") ?: JSONObject()
        val causalRecap = payload.optJSONObject("causal_recap") ?: JSONObject()
        val structuralLesson = payload.optJSONObject("structural_lesson") ?: JSONObject()
        val presentStateLine = payload.optJSONObject("present_state_line") ?: JSONObject()
        val postCommit = payload.optJSONObject("post_commit") ?: JSONObject()
        val cta = payload.optJSONObject("cta") ?: JSONObject()


        val resolutionProfile = payload.optString("resolution_profile")
        val isBasicResolution =
            resolutionProfile == "BASE_SINGLES_RESOLUTION" ||
                    resolutionProfile == "FULL_HOUSE_RESOLUTION"
        val isAdvancedResolution =
            resolutionProfile == "SUBSETS_RESOLUTION" ||
                    resolutionProfile == "INTERSECTIONS_RESOLUTION" ||
                    resolutionProfile == "ADVANCED_PATTERN_RESOLUTION"

        val out = JSONObject()
        out.put("schema_version", "resolution_reply_packet_projection_v3")
        out.put("resolution_profile", payload.opt("resolution_profile") ?: JSONObject.NULL)
        out.put("archetype", payload.opt("archetype") ?: JSONObject.NULL)

        out.put("technique", JSONObject().apply {
            put(
                "name",
                technique.opt("name")
                    ?: technique.opt("technique_name")
                    ?: JSONObject.NULL
            )
            put(
                "real_name",
                technique.opt("real_name")
                    ?: technique.opt("name")
                    ?: technique.opt("technique_name")
                    ?: JSONObject.NULL
            )
        })

        out.put("commit_truth", JSONObject().apply {
            put("cell", commit.opt("cell") ?: JSONObject.NULL)
            put("digit", commit.opt("digit") ?: JSONObject.NULL)
            put("authorized", commit.optBoolean("authorized", false))
            put(
                "present_state_language_required",
                commit.optBoolean("present_state_language_required", false)
            )
            put(
                "present_state_line",
                presentStateLine.opt("summary") ?: JSONObject.NULL
            )
        })

        out.put("compact_recap", JSONObject().apply {
            put("summary", recap.opt("summary") ?: JSONObject.NULL)
            put("max_beats", recap.opt("max_beats") ?: JSONObject.NULL)
        })

        out.put("honest_technique_contribution", JSONObject().apply {
            put("summary", techniqueContribution.opt("summary") ?: JSONObject.NULL)
            put(
                "final_forcing_summary",
                finalForcing.opt("summary") ?: JSONObject.NULL
            )
            put(
                "two_layer_honesty_line",
                honesty.opt("two_layer_honesty_line") ?: JSONObject.NULL
            )
            put(
                "must_distinguish_technique_from_finish",
                honesty.optBoolean("must_distinguish_technique_from_finish", false)
            )
        })


        out.put("causal_recap_surface", JSONObject().apply {
            put("battlefield_house", causalRecap.opt("battlefield_house") ?: JSONObject.NULL)
            put("source_house", causalRecap.opt("source_house") ?: JSONObject.NULL)
            put("source_overlap_cells", causalRecap.opt("source_overlap_cells") ?: JSONArray())
            put("final_removed_rival", causalRecap.opt("final_removed_rival") ?: JSONObject.NULL)
            put("ordinary_groundwork_line", causalRecap.opt("ordinary_groundwork_line") ?: JSONObject.NULL)
            put("decisive_cut_line", causalRecap.opt("decisive_cut_line") ?: JSONObject.NULL)
            put("survivor_line", causalRecap.opt("survivor_line") ?: JSONObject.NULL)
            put(
                "birthplace_vs_battleground_line",
                causalRecap.opt("birthplace_vs_battleground_line") ?: JSONObject.NULL
            )
        })

        out.put("lesson_surface", JSONObject().apply {
            put(
                "pressure_principle_line",
                structuralLesson.opt("pressure_principle_line") ?: JSONObject.NULL
            )
            put(
                "memory_rule_line",
                structuralLesson.opt("memory_rule_line") ?: JSONObject.NULL
            )
            put(
                "structural_insight_line",
                structuralLesson.opt("structural_insight_line") ?: JSONObject.NULL
            )
        })

        out.put("resolution_doctrine_surface", JSONObject().apply {
            put(
                "resolution_family",
                when {
                    isBasicResolution -> "BASIC"
                    isAdvancedResolution -> "ADVANCED"
                    else -> JSONObject.NULL
                }
            )
            put(
                "takeaway_focus",
                when {
                    isBasicResolution -> "observational_lesson"
                    isAdvancedResolution -> "structural_insight"
                    else -> JSONObject.NULL
                }
            )
            put(
                "closure_feel",
                when {
                    isBasicResolution -> "quiet_payoff"
                    isAdvancedResolution -> "elegant_payoff"
                    else -> "gentle_closure"
                }
            )
            put(
                "takeaway_sentence_starter",
                when {
                    isBasicResolution -> "The lesson here is"
                    isAdvancedResolution -> "The moral of this story is"
                    else -> JSONObject.NULL
                }
            )
            put(
                "observational_lesson_hint",
                if (isBasicResolution) {
                    "Sharpen the solver's eye: what quiet principle or scarcity should they notice next time?"
                } else {
                    JSONObject.NULL
                }
            )
            put(
                "principle_of_notice",
                if (isBasicResolution) {
                    techniqueContribution.opt("summary")
                        ?: finalForcing.opt("summary")
                        ?: recap.opt("summary")
                        ?: JSONObject.NULL
                } else {
                    JSONObject.NULL
                }
            )
            put(
                "structural_lesson_hint",
                if (isAdvancedResolution) {
                    "Explain how the pattern controlled the scene, not just what answer it produced."
                } else {
                    JSONObject.NULL
                }
            )
            put(
                "pattern_control_hint",
                if (isAdvancedResolution) {
                    honesty.opt("two_layer_honesty_line")
                        ?: techniqueContribution.opt("summary")
                        ?: finalForcing.opt("summary")
                        ?: recap.opt("summary")
                        ?: JSONObject.NULL
                } else {
                    JSONObject.NULL
                }
            )
            put(
                "graceful_exit_hint",
                "After the takeaway, let the technique leave the stage gracefully and hand off to the next step."
            )
        })

        out.put("post_commit_bridge", JSONObject().apply {
            put(
                "board_delta_summary",
                postCommit.opt("board_delta_summary") ?: JSONObject.NULL
            )
            put("placement_count", postCommit.opt("placement_count") ?: JSONObject.NULL)
        })

        out.put("cta", JSONObject().apply {
            put("kind", cta.opt("kind") ?: JSONObject.NULL)
        })

        out.put("support", JSONObject().apply {
            put("has_commit_truth", commit.has("cell") || commit.has("digit"))
            put("has_compact_recap", recap.has("summary"))
            put("has_honest_technique_contribution", techniqueContribution.has("summary"))
            put("has_causal_recap_surface", causalRecap.length() > 0)
            put("has_lesson_surface", structuralLesson.length() > 0)
            put("has_resolution_doctrine_surface", true)
            put("has_post_commit_bridge", postCommit.has("board_delta_summary"))
            put("has_cta_kind", cta.has("kind"))
            put("speech_surface_mode", "LEAN_STAGE_OWNER")
        })

        return out
    }




    fun projectGlossaryMini(replyRequest: ReplyRequestV1): JSONObject {
        val glossaryPayload = findFactPayload(replyRequest, FactBundleV1.Type.GLOSSARY_BUNDLE.name)
            ?: return JSONObject()
        val setupPacket =
            findAuthorizedFactPayload(replyRequest, FactBundleV1.Type.SETUP_REPLY_PACKET_V1)
        val terms = glossaryPayload.optJSONArray("terms") ?: JSONArray()
        val speechPolicy = glossaryPayload.optJSONObject("speech_policy") ?: JSONObject()
        val technique = setupPacket?.optJSONObject("technique") ?: JSONObject()
        val lens = setupPacket?.optJSONObject("lens") ?: JSONObject()
        val archetype = setupPacket?.optString("archetype")?.takeIf { it.isNotBlank() }

        fun firstMatchingTerm(vararg ids: String): JSONObject? {
            for (wanted in ids) {
                for (i in 0 until terms.length()) {
                    val t = terms.optJSONObject(i) ?: continue
                    if (t.optString("id") == wanted) return t
                }
            }
            return null
        }

        val coordinateSpeech = speechPolicy.optJSONObject("coordinate_speech") ?: JSONObject()
        val boxNaming = speechPolicy.optJSONObject("box_naming") ?: JSONObject()

        val relevantTerm = when (archetype) {
            "SUBSETS" -> firstMatchingTerm("subset_member", "subset_candidates", "pattern_cell", "pattern")
            "INTERSECTIONS" -> firstMatchingTerm("intersection", "pointing", "claim", "house_claim")
            "SINGLES", "NAKED_SINGLE", "HIDDEN_SINGLE" -> firstMatchingTerm("spotlight", "proof_beat", "commit")
            else -> firstMatchingTerm("archetype", "spotlight")
        }

        val out = JSONObject()
        out.put("schema_version", "glossary_mini_v1")

        if (technique.length() > 0) {
            out.put(
                "technique_player_name",
                technique.opt("name") ?: technique.opt("id") ?: JSONObject.NULL
            )
            out.put(
                "family_description_short",
                technique.opt("family") ?: lens.opt("summary") ?: JSONObject.NULL
            )
        }

        if (relevantTerm != null) {
            out.put("relevant_term", JSONObject().apply {
                put("id", relevantTerm.opt("id") ?: JSONObject.NULL)
                put("meaning", relevantTerm.opt("meaning") ?: JSONObject.NULL)
            })
        }

        out.put("say", JSONArray().apply {
            val coordinateDo = coordinateSpeech.optString("do")
            if (coordinateDo.isNotBlank()) put(coordinateDo)

            val boxDo = boxNaming.optString("do")
            if (boxDo.isNotBlank()) put(boxDo)
        })

        out.put("dont_say", JSONArray().apply {
            val coordinateDont = coordinateSpeech.optString("dont")
            if (coordinateDont.isNotBlank()) put(coordinateDont)

            val boxDont = boxNaming.optString("dont")
            if (boxDont.isNotBlank()) put(boxDont)
        })

        return if (out.length() > 1) out else JSONObject()
    }

    fun projectTechniqueCardMini(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.TEACHING_CARD_V1.name)
            ?: return JSONObject()

        val out = JSONObject()
        copyIfPresent(payload, out, "title")
        copyIfPresent(payload, out, "short_explanation")
        copyIfPresent(payload, out, "technique")
        copyIfPresent(payload, out, "definition")
        copyIfPresent(payload, out, "tip")
        copyIfPresent(payload, out, "synonyms")
        copyIfPresent(payload, out, "interesting_facts")
        copyIfPresent(payload, out, "notes")
        copyIfPresent(payload, out, "family_description")
        copyIfPresent(payload, out, "difficulty_level")
        return out
    }

    fun projectHandoverNoteMini(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.HANDOVER_NOTE_V1.name)
            ?: return JSONObject()

        val prev = payload.optJSONObject("prev") ?: JSONObject()
        val next = payload.optJSONObject("next") ?: JSONObject()
        val bridgeHints = payload.optJSONArray("bridge_hints") ?: JSONArray()

        val out = JSONObject()
        out.put(
            "previous_technique_short_label",
            prev.opt("real_name")
                ?: prev.opt("app_name")
                ?: prev.opt("family")
                ?: prev.opt("technique_id")
                ?: JSONObject.NULL
        )
        out.put(
            "next_technique_short_label",
            next.opt("real_name")
                ?: next.opt("app_name")
                ?: next.opt("family")
                ?: next.opt("technique_id")
                ?: JSONObject.NULL
        )
        out.put("relation", payload.opt("relation") ?: JSONObject.NULL)
        out.put(
            "bridge_hint",
            if (bridgeHints.length() > 0) bridgeHints.opt(0) else JSONObject.NULL
        )
        return out
    }

    fun projectOverlayMini(replyRequest: ReplyRequestV1): JSONObject {
        val payload = findFactPayload(replyRequest, FactBundleV1.Type.OVERLAY_FRAMES.name)
            ?: return JSONObject()

        val out = JSONObject()
        copyIfPresent(payload, out, "focus")
        copyIfPresent(payload, out, "frames")
        copyIfPresent(payload, out, "target_cell")
        copyIfPresent(payload, out, "target_house")
        return out
    }

    private fun repairForbiddenStoryPacketTypesV1(
        replyRequest: ReplyRequestV1
    ): JSONArray {
        val forbidden = setOf(
            FactBundleV1.Type.SOLVING_STEP_PACKET,
            FactBundleV1.Type.SOLVING_STEP_PACKET_V1,
            FactBundleV1.Type.SETUP_REPLY_PACKET_V1,
            FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1,
            FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1,
            FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1,
            FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1,
            FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1,
            FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1,
            FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1,
            FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1,
            FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1,
            FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
            FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1,
            FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
            FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
            FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1,
            FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1,
            FactBundleV1.Type.SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET_V1,
            FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1,
            FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1,
            FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1,
            FactBundleV1.Type.STORY_SIGNATURE_V1,
            FactBundleV1.Type.OVERLAY_FRAMES
        )

        return JSONArray().apply {
            replyRequest.facts
                .filter { it.type in forbidden }
                .map { it.type.name }
                .distinct()
                .forEach { put(it) }
        }
    }

    fun projectRepairContext(replyRequest: ReplyRequestV1): JSONObject {
        val t = replyRequest.turn
        val story = t.story
        val leakedTypes = repairForbiddenStoryPacketTypesV1(replyRequest)

        return JSONObject().apply {
            put("user_text", t.userText)
            put("pending_after", t.pendingAfter ?: JSONObject.NULL)
            put("recent_turns", takeFirstNRecentTurns(replyRequest.recentTurns, 3))
            put("decision_summary", replyRequest.decision.summary)
            put("story_stage_when_repair_started", story?.stage ?: JSONObject.NULL)
            put("step_id_when_repair_started", story?.stepId ?: JSONObject.NULL)
            put("grid_hash12_when_repair_started", story?.gridHash12 ?: JSONObject.NULL)
            put("repair_forbidden_story_packet_types_present", leakedTypes)
            put("repair_leak_present", leakedTypes.length() > 0)
            put(
                "repair_guidance",
                "Acknowledge the contradiction or loop, avoid introducing a fresh solving route, and only resume the same route if the user asks to continue."
            )
        }
    }



    // ---------------------------------------------------------------------
    // Phase 2 — detour packet projectors
    // ---------------------------------------------------------------------

    fun projectStepClarificationPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1.name)
            ?: JSONObject()

    fun projectProofChallengePacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1.name)
            ?: JSONObject()

    fun projectUserReasoningCheckPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1.name)
            ?: JSONObject()

    fun projectAlternativeTechniquePacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverCellCandidatesPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverCellsCandidatesPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverHouseCandidateMapPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverCellDigitBlockersPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverReasoningCheckPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1.name)
            ?: JSONObject()


    fun projectSolverAlternativeTechniquePacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverTechniqueScopeCheckPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverLocalMoveSearchPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverRouteComparisonPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1.name)
            ?: JSONObject()

    fun projectSolverScopedSupportPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1.name)
            ?: JSONObject()

    fun projectTargetCellQueryPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1.name)
            ?: JSONObject()

    fun projectCandidateStatePacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1.name)
            ?: JSONObject()

    fun projectNeighborCellQueryPacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1.name)
            ?: JSONObject()

    fun projectReturnToRoutePacket(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1.name)
            ?: JSONObject()

    fun projectNormalizedDetourMoveProof(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.NORMALIZED_DETOUR_MOVE_PROOF_V1.name)
            ?.optJSONObject("result")
            ?: JSONObject()

    fun projectNormalizedDetourLocalInspection(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.NORMALIZED_DETOUR_LOCAL_INSPECTION_V1.name)
            ?.optJSONObject("result")
            ?: JSONObject()

    fun projectNormalizedDetourProposalVerdict(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.NORMALIZED_DETOUR_PROPOSAL_VERDICT_V1.name)
            ?.optJSONObject("result")
            ?: JSONObject()

    fun projectDetourNarrativeContext(replyRequest: ReplyRequestV1): JSONObject =
        findFactPayload(replyRequest, FactBundleV1.Type.DETOUR_NARRATIVE_CONTEXT_V1.name)
            ?: JSONObject()

    private fun firstNonBlankStringV1(vararg values: String?): String? {
        values.forEach { value ->
            val cleaned = value?.trim()
            if (!cleaned.isNullOrEmpty()) return cleaned
        }
        return null
    }

    private fun firstJsonObjectOrEmptyV1(vararg values: JSONObject?): JSONObject {
        values.forEach { value ->
            if (value != null && value.length() > 0) return value
        }
        return JSONObject()
    }

    private fun jsonArrayOfStringsV1(values: List<String>): JSONArray =
        JSONArray().apply { values.forEach { put(it) } }

    private fun detourAnswerBoundaryJsonV1(vararg values: DetourAnswerBoundaryV1): JSONArray =
        JSONArray().apply { values.forEach { put(it.name) } }

    private fun detourNarrativeSurfaceV1(
        replyRequest: ReplyRequestV1,
        doctrineFamily: String,
        defaultAnswerShape: String,
        defaultOrderedExplanationLadder: List<String>,
        defaultBoundaryLine: String,
        defaultHandbackLine: String
    ): JSONObject {

        val narrativeContext = projectDetourNarrativeContext(replyRequest)
        val dominantAtom = narrativeContext.optJSONObject("dominant_atom") ?: JSONObject()
        val handback = narrativeContext.optJSONObject("handback_policy") ?: JSONObject()

        val moveProofPacket = projectDetourMoveProofPacket(replyRequest)
        val proofGeometryKind =
            firstNonBlankStringV1(
                moveProofPacket.optJSONObject("local_proof_geometry")?.optString("geometry_kind", null)
            )?.trim()?.uppercase()

        val rawAtomKind = dominantAtom.optString("atom_kind", "").trim().uppercase()

        val atomKind =
            if (
                doctrineFamily.contains("proof", ignoreCase = true) &&
                rawAtomKind == "STATE_READOUT" &&
                !proofGeometryKind.isNullOrBlank() &&
                proofGeometryKind != "NONE"
            ) {
                "LOCAL_PROOF_SPOTLIGHT"
            } else {
                rawAtomKind
            }

        val orderedLadder = when (atomKind) {
            "LOCAL_PROOF_SPOTLIGHT" -> listOf(
                "focus",
                "decisive_fact",
                "evidence",
                "consequence",
                "bounded_handback"
            )

            "STATE_READOUT" -> listOf(
                "scope",
                "state_readout",
                "why_it_matters",
                "bounded_handback"
            )

            "PROPOSAL_VERDICT" -> listOf(
                "verdict",
                "what_works",
                "what_fails_or_is_missing",
                "route_relation",
                "bounded_handback"
            )

            else -> defaultOrderedExplanationLadder
        }

        val answerShape = when (atomKind) {
            "LOCAL_PROOF_SPOTLIGHT" ->
                "focus -> decisive fact -> bounded evidence -> local consequence -> bounded handback"

            "STATE_READOUT" ->
                "scope -> state readout -> why it matters -> bounded handback"

            "PROPOSAL_VERDICT" ->
                "verdict -> what works -> what fails or is missing -> route relation -> bounded handback"

            else -> defaultAnswerShape
        }

        val boundaryLine = when (atomKind) {
            "LOCAL_PROOF_SPOTLIGHT" ->
                "Stop once the local proof answer is complete. Do not reopen the whole step or commit the move."

            "STATE_READOUT" ->
                "Keep this as a bounded local readout. Do not expand into a full proof ladder."

            "PROPOSAL_VERDICT" ->
                "Give the verdict and bounded reasoning only. Do not switch routes or turn this into a board audit."

            else -> defaultBoundaryLine
        }

        val handbackLine = firstNonBlankStringV1(
            dominantAtom.optString("spoken_return_line", null),
            handback.optString("spoken_return_line", null),
            defaultHandbackLine
        )

        return JSONObject().apply {
            put(
                "doctrine_surface",
                JSONObject().apply {
                    put("family", doctrineFamily)
                    put("native_context_present", narrativeContext.length() > 0)
                    put(
                        "native_archetype",
                        firstNonBlankStringV1(
                            narrativeContext.optString("archetype", null),
                            dominantAtom.optString("archetype", null)
                        ) ?: JSONObject.NULL
                    )
                    put("answer_priority", orderedLadder.firstOrNull() ?: JSONObject.NULL)
                    put(
                        "must_not",
                        jsonArrayOfStringsV1(
                            listOf(
                                "switch_route",
                                "reopen_full_step",
                                "exceed_detour_boundary"
                            )
                        )
                    )
                }
            )
            put("answer_shape", answerShape)
            put("ordered_explanation_ladder", jsonArrayOfStringsV1(orderedLadder))
            put("boundary_line", boundaryLine)
            put("handback_line", handbackLine ?: JSONObject.NULL)
        }
    }

    fun projectDetourAlternativeTechniquePacket(replyRequest: ReplyRequestV1): JSONObject {
        val detourAlternative =
            findFactPayload(replyRequest, FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
                ?: JSONObject()
        val solverAlternative =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
                ?: JSONObject()
        val solverRouteComparison =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1.name)
                ?: JSONObject()
        val solverScopedSupport =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1.name)
                ?: JSONObject()
        val solvingStep =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)
                ?: JSONObject()

        if (
            detourAlternative.length() == 0 &&
            solverAlternative.length() == 0 &&
            solverRouteComparison.length() == 0
        ) {
            return JSONObject()
        }

        val focusCells = buildList {
            firstNonBlankStringV1(
                detourAlternative.optString("target_cell", null),
                solverAlternative.optString("target_cell", null),
                solvingStep.optString("target_cell", null)
            )?.let { add(it) }
        }.distinct()

        val focusHouses = buildList {
            firstNonBlankStringV1(
                detourAlternative.optString("house_scope", null),
                solverAlternative.optString("house_scope", null),
                solverRouteComparison.optString("house_scope", null)
            )?.let { add(it) }
        }.distinct()

        val alternativeNarrativeSurface =
            detourNarrativeSurfaceV1(
                replyRequest = replyRequest,
                doctrineFamily = "alternative_technique_doctrine",
                defaultAnswerShape = "current route -> asked alternative -> fit or not -> solver preference -> bounded handback",
                defaultOrderedExplanationLadder = listOf(
                    "current_route",
                    "asked_alternative",
                    "fit_or_not",
                    "solver_preference",
                    "bounded_handback"
                ),
                defaultBoundaryLine = "Keep this comparative and route-aware. Do not switch routes or reopen the full solve.",
                defaultHandbackLine = "That answers the alternative-technique question without changing our paused route."
            )

        return JSONObject().apply {
            put("packet_kind", "detour_alternative_technique")
            put(
                "profile",
                when {
                    solverRouteComparison.length() > 0 -> "alternative_route_comparison"
                    solverAlternative.length() > 0 -> "alternative_technique_solver_check"
                    else -> "alternative_technique_detour"
                }
            )
            put(
                "anchor_step_id",
                firstNonBlankStringV1(
                    detourAlternative.optString("step_id", null),
                    solverAlternative.optString("step_id", null),
                    solverRouteComparison.optString("step_id", null),
                    solvingStep.optString("step_id", null)
                ) ?: JSONObject.NULL
            )
            put(
                "anchor_story_stage",
                firstNonBlankStringV1(
                    detourAlternative.optString("story_stage", null),
                    solverAlternative.optString("story_stage", null),
                    solverRouteComparison.optString("story_stage", null),
                    replyRequest.turn.story?.stage
                ) ?: JSONObject.NULL
            )
            put(
                "current_technique",
                firstJsonObjectOrEmptyV1(
                    solverRouteComparison.optJSONObject("current_route_technique"),
                    solverAlternative.optJSONObject("current_route_technique"),
                    solvingStep.optJSONObject("technique")
                )
            )
            put(
                "asked_technique",
                firstJsonObjectOrEmptyV1(
                    detourAlternative.optJSONObject("asked_technique"),
                    solverAlternative.optJSONObject("asked_technique"),
                    solverRouteComparison.optJSONObject("asked_route_technique")
                )
            )
            put(
                "availability_verdict",
                firstNonBlankStringV1(
                    solverAlternative.optString("availability_verdict", null),
                    solverRouteComparison.optString("comparison_verdict", null),
                    detourAlternative.optString("availability_verdict", null)
                ) ?: JSONObject.NULL
            )
            put(
                "why_current_route_chosen",
                firstNonBlankStringV1(
                    solverRouteComparison.optString("why_current_route_chosen", null),
                    solverAlternative.optString("why_current_route_chosen", null),
                    detourAlternative.optString("why_current_route_chosen", null)
                ) ?: JSONObject.NULL
            )
            put(
                "why_alternative_does_or_does_not_fit",
                firstNonBlankStringV1(
                    solverAlternative.optString("why_alternative_does_or_does_not_fit", null),
                    solverAlternative.optString("reason", null),
                    solverRouteComparison.optString("why_routes_differ", null),
                    detourAlternative.optString("question", null)
                ) ?: JSONObject.NULL
            )
            put(
                "truth_scope",
                firstNonBlankStringV1(
                    detourAlternative.optString("scope_kind", null),
                    solverAlternative.optString("scope_kind", null),
                    solverRouteComparison.optString("scope_kind", null),
                    detourAlternative.optString("house_scope", null)
                ) ?: JSONObject.NULL
            )
            put(
                "doctrine_surface",
                alternativeNarrativeSurface.optJSONObject("doctrine_surface") ?: JSONObject()
            )
            put(
                "answer_shape",
                alternativeNarrativeSurface.optString("answer_shape", "")
                    .ifBlank { JSONObject.NULL as String? } ?: JSONObject.NULL
            )
            put(
                "ordered_explanation_ladder",
                alternativeNarrativeSurface.optJSONArray("ordered_explanation_ladder") ?: JSONArray()
            )
            put(
                "boundary_line",
                firstNonBlankStringV1(
                    alternativeNarrativeSurface.optString("boundary_line", null)
                ) ?: JSONObject.NULL
            )
            put(
                "handback_line",
                firstNonBlankStringV1(
                    alternativeNarrativeSurface.optString("handback_line", null)
                ) ?: JSONObject.NULL
            )
            put(
                "support",
                JSONObject().apply {
                    if (detourAlternative.length() > 0) put("alternative_technique_packet", detourAlternative)
                    if (solverAlternative.length() > 0) put("solver_alternative_technique_packet", solverAlternative)
                    if (solverRouteComparison.length() > 0) put("solver_route_comparison_packet", solverRouteComparison)
                    if (solverScopedSupport.length() > 0) put("solver_scoped_support_packet", solverScopedSupport)
                }
            )
            put(
                "overlay_policy",
                projectWave1DetourOverlayPolicyV1(
                    focusCells = focusCells,
                    focusHouses = focusHouses,
                    reasonForFocus = "alternative_technique_detour",
                    spokenAnchor = focusCells.firstOrNull() ?: focusHouses.firstOrNull()
                ).toJson()
            )
            put(
                "handback_policy",
                projectWave1HandbackPolicyV1(
                    replyRequest,
                    spokenReturnLine = "That answers the alternative-technique question without changing our paused route."
                ).toJson()
            )
            put(
                "answer_boundary",
                detourAnswerBoundaryJsonV1(
                    DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                    DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
                    DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
                )
            )
        }
    }

    fun projectDetourLocalMoveSearchPacket(replyRequest: ReplyRequestV1): JSONObject {
        val solverLocalMoveSearch =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1.name)
                ?: JSONObject()
        val solverCellCandidates =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1.name)
                ?: JSONObject()
        val solverCellsCandidates =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1.name)
                ?: JSONObject()
        val solverHouseMap =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1.name)
                ?: JSONObject()
        val solverScopedSupport =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1.name)
                ?: JSONObject()

        if (solverLocalMoveSearch.length() == 0) {
            return JSONObject()
        }

        val focusCells = buildList {
            firstNonBlankStringV1(
                solverLocalMoveSearch.optString("target_cell", null),
                solverLocalMoveSearch.optString("focus_cell", null),
                solverCellCandidates.optString("cell", null)
            )?.let { add(it) }
        }.distinct()

        val focusHouses = buildList {
            firstNonBlankStringV1(
                solverLocalMoveSearch.optString("house_scope", null),
                solverHouseMap.optString("house_scope", null),
                solverLocalMoveSearch.optString("scope_ref", null)
            )?.let { add(it) }
        }.distinct()

        val localMoveNarrativeSurface =
            detourNarrativeSurfaceV1(
                replyRequest = replyRequest,
                doctrineFamily = "local_move_search_doctrine",
                defaultAnswerShape = "scope -> local options -> found move or no move -> route relation -> bounded handback",
                defaultOrderedExplanationLadder = listOf(
                    "scope",
                    "local_options",
                    "found_move_or_no_move",
                    "route_relation",
                    "bounded_handback"
                ),
                defaultBoundaryLine = "Keep this as bounded local search only. Do not widen into whole-grid solving or switch routes.",
                defaultHandbackLine = "That answers the local move-search question while keeping the original paused move intact."
            )

        return JSONObject().apply {
            put("packet_kind", "detour_local_move_search")
            put("profile", "local_move_search")
            put(
                "anchor_step_id",
                firstNonBlankStringV1(
                    solverLocalMoveSearch.optString("step_id", null)
                ) ?: JSONObject.NULL
            )
            put(
                "anchor_story_stage",
                firstNonBlankStringV1(
                    solverLocalMoveSearch.optString("story_stage", null),
                    replyRequest.turn.story?.stage
                ) ?: JSONObject.NULL
            )
            put(
                "search_scope",
                firstNonBlankStringV1(
                    solverLocalMoveSearch.optString("scope_kind", null),
                    solverLocalMoveSearch.optString("scope_ref", null),
                    solverLocalMoveSearch.optString("house_scope", null)
                ) ?: JSONObject.NULL
            )
            put(
                "local_candidates",
                firstJsonObjectOrEmptyV1(
                    solverLocalMoveSearch.optJSONObject("local_candidates"),
                    solverCellCandidates,
                    solverCellsCandidates,
                    solverHouseMap
                )
            )
            put(
                "found_move_summary",
                firstJsonObjectOrEmptyV1(
                    solverLocalMoveSearch.optJSONObject("found_move_summary"),
                    solverLocalMoveSearch.optJSONObject("best_local_move"),
                    solverLocalMoveSearch.optJSONObject("move_summary")
                )
            )
            put(
                "no_move_reason",
                firstNonBlankStringV1(
                    solverLocalMoveSearch.optString("no_move_reason", null),
                    solverLocalMoveSearch.optString("why_no_local_move", null)
                ) ?: JSONObject.NULL
            )
            put(
                "route_relationship",
                firstNonBlankStringV1(
                    solverLocalMoveSearch.optString("route_relationship", null),
                    "bounded_local_search_without_route_switch"
                ) ?: JSONObject.NULL
            )
            put(
                "doctrine_surface",
                localMoveNarrativeSurface.optJSONObject("doctrine_surface") ?: JSONObject()
            )
            put(
                "answer_shape",
                firstNonBlankStringV1(localMoveNarrativeSurface.optString("answer_shape", null))
                    ?: JSONObject.NULL
            )
            put(
                "ordered_explanation_ladder",
                localMoveNarrativeSurface.optJSONArray("ordered_explanation_ladder") ?: JSONArray()
            )
            put(
                "boundary_line",
                firstNonBlankStringV1(localMoveNarrativeSurface.optString("boundary_line", null))
                    ?: JSONObject.NULL
            )
            put(
                "handback_line",
                firstNonBlankStringV1(localMoveNarrativeSurface.optString("handback_line", null))
                    ?: JSONObject.NULL
            )
            put(
                "support",
                JSONObject().apply {
                    put("solver_local_move_search_packet", solverLocalMoveSearch)
                    if (solverCellCandidates.length() > 0) put("solver_cell_candidates_packet", solverCellCandidates)
                    if (solverCellsCandidates.length() > 0) put("solver_cells_candidates_packet", solverCellsCandidates)
                    if (solverHouseMap.length() > 0) put("solver_house_candidate_map_packet", solverHouseMap)
                    if (solverScopedSupport.length() > 0) put("solver_scoped_support_packet", solverScopedSupport)
                }
            )
            put(
                "overlay_policy",
                projectWave1DetourOverlayPolicyV1(
                    focusCells = focusCells,
                    focusHouses = focusHouses,
                    reasonForFocus = "local_move_search_detour",
                    spokenAnchor = focusCells.firstOrNull() ?: focusHouses.firstOrNull()
                ).toJson()
            )
            put(
                "handback_policy",
                projectWave1HandbackPolicyV1(
                    replyRequest,
                    spokenReturnLine = "That answers the local move-search question while keeping the original paused move intact."
                ).toJson()
            )
            put(
                "answer_boundary",
                detourAnswerBoundaryJsonV1(
                    DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                    DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
                    DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
                )
            )
        }
    }

    fun projectDetourRouteComparisonPacket(replyRequest: ReplyRequestV1): JSONObject {
        val solverRouteComparison =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1.name)
                ?: JSONObject()
        val solverAlternative =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
                ?: JSONObject()
        val detourAlternative =
            findFactPayload(replyRequest, FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1.name)
                ?: JSONObject()
        val solverScopedSupport =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1.name)
                ?: JSONObject()
        val solvingStep =
            findFactPayload(replyRequest, FactBundleV1.Type.SOLVING_STEP_PACKET_V1.name)
                ?: JSONObject()

        if (solverRouteComparison.length() == 0) {
            return JSONObject()
        }

        val focusCells = buildList {
            firstNonBlankStringV1(
                solverRouteComparison.optString("target_cell", null),
                solvingStep.optString("target_cell", null)
            )?.let { add(it) }
        }.distinct()

        val focusHouses = buildList {
            firstNonBlankStringV1(
                solverRouteComparison.optString("house_scope", null),
                solverRouteComparison.optString("scope_ref", null)
            )?.let { add(it) }
        }.distinct()

        val routeComparisonNarrativeSurface =
            detourNarrativeSurfaceV1(
                replyRequest = replyRequest,
                doctrineFamily = "route_comparison_doctrine",
                defaultAnswerShape = "current route -> asked route -> relation -> solver preference -> bounded handback",
                defaultOrderedExplanationLadder = listOf(
                    "current_route",
                    "asked_route",
                    "relation",
                    "solver_preference",
                    "bounded_handback"
                ),
                defaultBoundaryLine = "Keep this as route comparison only. Do not turn it into a proof challenge or switch routes.",
                defaultHandbackLine = "That compares the route options without changing the solver’s paused route."
            )

        return JSONObject().apply {
            put("packet_kind", "detour_route_comparison")
            put("profile", "route_comparison")
            put(
                "anchor_step_id",
                firstNonBlankStringV1(
                    solverRouteComparison.optString("step_id", null),
                    solvingStep.optString("step_id", null)
                ) ?: JSONObject.NULL
            )
            put(
                "anchor_story_stage",
                firstNonBlankStringV1(
                    solverRouteComparison.optString("story_stage", null),
                    replyRequest.turn.story?.stage
                ) ?: JSONObject.NULL
            )
            put(
                "current_route_summary",
                firstJsonObjectOrEmptyV1(
                    solverRouteComparison.optJSONObject("current_route_summary"),
                    solverRouteComparison.optJSONObject("current_route"),
                    solvingStep
                )
            )
            put(
                "asked_route_summary",
                firstJsonObjectOrEmptyV1(
                    solverRouteComparison.optJSONObject("asked_route_summary"),
                    solverRouteComparison.optJSONObject("alternative_route"),
                    detourAlternative.optJSONObject("asked_technique"),
                    solverAlternative.optJSONObject("asked_technique")
                )
            )
            put(
                "equivalence_or_difference",
                firstNonBlankStringV1(
                    solverRouteComparison.optString("equivalence_or_difference", null),
                    solverRouteComparison.optString("comparison_verdict", null),
                    solverRouteComparison.optString("relation", null)
                ) ?: JSONObject.NULL
            )
            put(
                "why_solver_prefers_current_route",
                firstNonBlankStringV1(
                    solverRouteComparison.optString("why_current_route_chosen", null),
                    solverRouteComparison.optString("why_solver_prefers_current_route", null)
                ) ?: JSONObject.NULL
            )
            put(
                "proof_boundedness",
                firstNonBlankStringV1(
                    solverRouteComparison.optString("proof_boundedness", null),
                    "comparison_only_no_route_switch"
                ) ?: JSONObject.NULL
            )
            put(
                "doctrine_surface",
                routeComparisonNarrativeSurface.optJSONObject("doctrine_surface") ?: JSONObject()
            )
            put(
                "answer_shape",
                firstNonBlankStringV1(routeComparisonNarrativeSurface.optString("answer_shape", null))
                    ?: JSONObject.NULL
            )
            put(
                "ordered_explanation_ladder",
                routeComparisonNarrativeSurface.optJSONArray("ordered_explanation_ladder") ?: JSONArray()
            )
            put(
                "boundary_line",
                firstNonBlankStringV1(routeComparisonNarrativeSurface.optString("boundary_line", null))
                    ?: JSONObject.NULL
            )
            put(
                "handback_line",
                firstNonBlankStringV1(routeComparisonNarrativeSurface.optString("handback_line", null))
                    ?: JSONObject.NULL
            )
            put(
                "support",
                JSONObject().apply {
                    put("solver_route_comparison_packet", solverRouteComparison)
                    if (solverAlternative.length() > 0) put("solver_alternative_technique_packet", solverAlternative)
                    if (detourAlternative.length() > 0) put("alternative_technique_packet", detourAlternative)
                    if (solverScopedSupport.length() > 0) put("solver_scoped_support_packet", solverScopedSupport)
                }
            )
            put(
                "overlay_policy",
                projectWave1DetourOverlayPolicyV1(
                    focusCells = focusCells,
                    focusHouses = focusHouses,
                    reasonForFocus = "route_comparison_detour",
                    spokenAnchor = focusCells.firstOrNull() ?: focusHouses.firstOrNull()
                ).toJson()
            )
            put(
                "handback_policy",
                projectWave1HandbackPolicyV1(
                    replyRequest,
                    spokenReturnLine = "That compares the route options without changing the solver’s paused route."
                ).toJson()
            )
            put(
                "answer_boundary",
                detourAnswerBoundaryJsonV1(
                    DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                    DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
                    DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
                )
            )
        }
    }

    private fun parseReasoningVerdictV1(raw: String?): ReasoningVerdictV1 {
        return when (raw?.trim()?.uppercase()) {
            "VALID" -> ReasoningVerdictV1.VALID
            "INVALID" -> ReasoningVerdictV1.INVALID
            "PARTIALLY_VALID" -> ReasoningVerdictV1.PARTIALLY_VALID
            "VALID_BUT_NOT_CURRENT_ROUTE" -> ReasoningVerdictV1.VALID_BUT_NOT_CURRENT_ROUTE
            else -> ReasoningVerdictV1.UNKNOWN
        }
    }

    private fun stringListFromJsonArrayV1(arr: JSONArray?): List<String> {
        if (arr == null) return emptyList()
        val out = mutableListOf<String>()
        for (i in 0 until arr.length()) {
            val s = arr.optString(i, "").trim()
            if (s.isNotEmpty()) out.add(s)
        }
        return out
    }

    private fun projectWave1DetourOverlayPolicyV1(
        focusCells: List<String> = emptyList(),
        focusHouses: List<String> = emptyList(),
        reasonForFocus: String? = null,
        spokenAnchor: String? = null
    ): DetourOverlayPolicyV1 {
        return DetourOverlayPolicyV1(
            overlayMode = if (focusCells.isNotEmpty() || focusHouses.isNotEmpty()) {
                DetourOverlayModeV1.REPLACE
            } else {
                DetourOverlayModeV1.PRESERVE
            },
            primaryFocusCells = focusCells,
            primaryFocusHouses = focusHouses,
            reasonForFocus = reasonForFocus,
            expectedSpokenAnchor = spokenAnchor
        )
    }

    private fun projectWave1HandbackPolicyV1(
        replyRequest: ReplyRequestV1,
        spokenReturnLine: String
    ): DetourHandbackPolicyV1 {
        val routeReturn = projectDetourRouteReturnPacket(replyRequest)
        val routeId = routeReturn.optString("route_id", "").trim().ifEmpty { null }
        val storyStage = routeReturn.optString("story_stage", "").trim().ifEmpty { null }
        val stepId = routeReturn.optString("step_id", "").trim().ifEmpty { null }
        val canReturn = replyRequest.turn.turnRouteReturnAllowed

        return DetourHandbackPolicyV1(
            handoverMode = if (canReturn) {
                DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
            } else {
                DetourHandoverModeV1.AWAIT_USER_CONTROL
            },
            pausedRouteCheckpoint = routeId,
            returnTargetStage = storyStage,
            returnTargetStepId = stepId,
            stayDetachedUntilUserSaysContinue = !canReturn,
            spokenReturnLine = spokenReturnLine
        )
    }

    private fun parseDetourHandoverModeV1(raw: String?): DetourHandoverModeV1 {
        return when (raw?.trim()?.uppercase()) {
            "RETURN_TO_CURRENT_STAGE" -> DetourHandoverModeV1.RETURN_TO_CURRENT_STAGE
            "RETURN_TO_CURRENT_MOVE" -> DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
            "HOLD_FOCUS_HERE" -> DetourHandoverModeV1.HOLD_FOCUS_HERE
            "AWAIT_USER_CONTROL" -> DetourHandoverModeV1.AWAIT_USER_CONTROL
            "REPAIR_THEN_RETURN" -> DetourHandoverModeV1.REPAIR_THEN_RETURN
            else -> DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE
        }
    }

    fun projectDetourMoveProofPacket(replyRequest: ReplyRequestV1): JSONObject {
        fun archetypeForProof(
            methodFamily: String?,
            proofObject: String?,
            challengeLane: String?,
            answerPolarity: String?,
            nonproofReason: String?,
            rivalCell: String?,
            claimedTechniqueId: String?
        ): String {
            val mf = methodFamily?.trim()?.uppercase()
            val po = proofObject?.trim()?.uppercase()
            val lane = challengeLane?.trim()?.uppercase()
            val polarity = answerPolarity?.trim()?.uppercase()
            val reason = nonproofReason?.trim()?.uppercase()
            val hasRivalCell = !rivalCell.isNullOrBlank()
            val hasClaimedTechnique = !claimedTechniqueId.isNullOrBlank()

            return when {
                mf == "TECHNIQUE_LEGITIMACY" || lane == "TECHNIQUE_LEGITIMACY" || hasClaimedTechnique ->
                    "PATTERN_LEGITIMACY_CHECK"

                mf == "CONTRAST_TEST" || po == "CELL_A_WINS_OVER_CELL_B_FOR_DIGIT" || hasRivalCell ->
                    "CONTRAST_DUEL"

                mf == "HOUSE_UNIQUENESS" || mf == "RIVAL_ELIMINATION_LADDER" ||
                        po == "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE" ||
                        po == "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL" ||
                        po == "CELL_CAN_BE_DIGIT" ->
                    "SURVIVOR_LADDER"

                mf == "DIRECT_CONTRADICTION" || mf == "ACTION_LEGITIMACY" ||
                        po == "CELL_CANNOT_BE_DIGIT" ||
                        po == "HOUSE_BLOCKS_DIGIT_FOR_TARGET" ||
                        po == "ELIMINATION_IS_LEGAL" ->
                    "LOCAL_CONTRADICTION_SPOTLIGHT"

                polarity == "NOT_LOCALLY_PROVED" -> {
                    if ((reason == "ROUTE_DEPENDENT_NOT_LOCALLY_VISIBLE" ||
                                reason == "ELIMINATION_SUPPORT_NOT_LOCALLY_VISIBLE") &&
                        hasClaimedTechnique
                    ) {
                        "PATTERN_LEGITIMACY_CHECK"
                    } else {
                        "HONEST_INSUFFICIENCY_ANSWER"
                    }
                }

                lane == "FORCEDNESS_OR_UNIQUENESS" ->
                    "SURVIVOR_LADDER"

                lane == "CANDIDATE_IMPOSSIBILITY" || lane == "ELIMINATION_LEGITIMACY" ->
                    "LOCAL_CONTRADICTION_SPOTLIGHT"

                else ->
                    "HONEST_INSUFFICIENCY_ANSWER"
            }
        }

        fun doctrineForArchetype(archetype: String): String =
            when (archetype.trim().uppercase()) {
                "LOCAL_CONTRADICTION_SPOTLIGHT" -> "contradiction_spotlight_v1"
                "SURVIVOR_LADDER" -> "survivor_ladder_v1"
                "CONTRAST_DUEL" -> "contrast_duel_v1"
                "PATTERN_LEGITIMACY_CHECK" -> "pattern_legitimacy_v1"
                else -> "honest_insufficiency_v1"
            }

        fun proofScopeFor(methodFamily: String?, nonproofReason: String?): String =
            when {
                nonproofReason?.trim()?.uppercase() == "ROUTE_DEPENDENT_NOT_LOCALLY_VISIBLE" ->
                    "LOCAL_BUT_ROUTE_DEPENDENT"

                methodFamily?.trim()?.uppercase() == "HOUSE_UNIQUENESS" ->
                    "HOUSE_LOCAL"

                methodFamily?.trim()?.uppercase() == "TECHNIQUE_LEGITIMACY" ->
                    "TECHNIQUE_LOCAL"

                else ->
                    "CELL_LOCAL"
            }

        fun proofStrengthFor(methodFamily: String?, answerPolarity: String?): String =
            when {
                answerPolarity?.trim()?.uppercase() == "NOT_LOCALLY_PROVED" ->
                    "NEGATIVE_HONESTY"

                methodFamily?.trim()?.uppercase() == "DIRECT_CONTRADICTION" ->
                    "DIRECT"

                else ->
                    "STRUCTURED"
            }

        fun packetArchetypeEnumV1(archetype: String): DetourNarrativeArchetypeV1 =
            when (archetype.trim().uppercase()) {
                "LOCAL_CONTRADICTION_SPOTLIGHT" -> DetourNarrativeArchetypeV1.LOCAL_CONTRADICTION_SPOTLIGHT
                "SURVIVOR_LADDER" -> DetourNarrativeArchetypeV1.SURVIVOR_LADDER
                "CONTRAST_DUEL" -> DetourNarrativeArchetypeV1.CONTRAST_DUEL
                "PATTERN_LEGITIMACY_CHECK" -> DetourNarrativeArchetypeV1.PATTERN_LEGITIMACY_CHECK
                "HONEST_INSUFFICIENCY_ANSWER" -> DetourNarrativeArchetypeV1.HONEST_INSUFFICIENCY_ANSWER
                else -> DetourNarrativeArchetypeV1.LOCAL_PROOF_SPOTLIGHT
            }

        fun parseCellRefV1(cellRef: String?): Pair<Int, Int>? {
            val m = Regex("""r([1-9])c([1-9])""", RegexOption.IGNORE_CASE)
                .matchEntire(cellRef?.trim().orEmpty()) ?: return null
            return m.groupValues[1].toInt() to m.groupValues[2].toInt()
        }

        fun parseHouseRefV1(houseRef: String?): Pair<String, Int>? {
            val m = Regex("""(row|col|box)\s*([1-9])""", RegexOption.IGNORE_CASE)
                .find(houseRef?.trim().orEmpty()) ?: return null
            return m.groupValues[1].lowercase() to m.groupValues[2].toInt()
        }

        fun houseContainsCellV1(houseRef: String?, cellRef: String?): Boolean {
            val cell = parseCellRefV1(cellRef) ?: return false
            val house = parseHouseRefV1(houseRef) ?: return false
            val row = cell.first
            val col = cell.second
            return when (house.first) {
                "row" -> row == house.second
                "col" -> col == house.second
                "box" -> ((row - 1) / 3) * 3 + ((col - 1) / 3) + 1 == house.second
                else -> false
            }
        }

        fun sortCellsByGridOrderV1(cells: Collection<String>): List<String> =
            cells
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .distinct()
                .sortedWith(
                    compareBy<String>(
                        { parseCellRefV1(it)?.first ?: 99 },
                        { parseCellRefV1(it)?.second ?: 99 }
                    )
                )

        fun absorbCellDigitsFromObjectV1(
            obj: JSONObject?,
            out: MutableMap<String, MutableSet<Int>>
        ) {
            if (obj == null || obj.length() == 0) return

            val cell =
                firstNonBlankStringV1(
                    obj.optString("cell", null),
                    obj.optString("cell_ref", null),
                    obj.optString("ref", null)
                ) ?: return

            val digits = mutableSetOf<Int>()

            fun absorbArray(arr: JSONArray?) {
                if (arr == null) return
                for (i in 0 until arr.length()) {
                    val d = arr.optInt(i, -1)
                    if (d in 1..9) digits += d
                }
            }

            absorbArray(obj.optJSONArray("digits"))
            absorbArray(obj.optJSONArray("candidates"))
            absorbArray(obj.optJSONArray("candidate_digits"))
            absorbArray(obj.optJSONArray("remaining_digits"))

            val directDigit = obj.optInt("digit", -1)
            if (directDigit in 1..9) digits += directDigit

            val directCandidateDigit = obj.optInt("candidate_digit", -1)
            if (directCandidateDigit in 1..9) digits += directCandidateDigit

            if (digits.isEmpty()) return
            out.getOrPut(cell) { linkedSetOf() }.addAll(digits)
        }

        fun extractCellDigitsMapFromCellsCandidatesV1(packet: JSONObject): Map<String, List<Int>> {
            val root = packet.optJSONObject("result") ?: packet
            val out = linkedMapOf<String, MutableSet<Int>>()

            absorbCellDigitsFromObjectV1(root, out)

            listOf("cells", "cell_candidates", "candidate_cells", "items", "results").forEach { key ->
                val arr = root.optJSONArray(key) ?: return@forEach
                for (i in 0 until arr.length()) {
                    absorbCellDigitsFromObjectV1(arr.optJSONObject(i), out)
                }
            }

            return out.mapValues { (_, digits) -> digits.toList().sorted() }
        }

        fun absorbDigitCellsObjectV1(
            obj: JSONObject?,
            requestedDigit: Int,
            out: MutableSet<String>
        ) {
            if (obj == null || obj.length() == 0) return

            fun absorbCellArray(arr: JSONArray?) {
                if (arr == null) return
                for (i in 0 until arr.length()) {
                    val cell =
                        firstNonBlankStringV1(
                            arr.optString(i, null),
                            arr.optJSONObject(i)?.optString("cell", null),
                            arr.optJSONObject(i)?.optString("cell_ref", null),
                            arr.optJSONObject(i)?.optString("ref", null)
                        )
                    if (!cell.isNullOrBlank()) out += cell
                }
            }

            absorbCellArray(obj.optJSONArray(requestedDigit.toString()))

            obj.optJSONObject(requestedDigit.toString())?.let { digitObj ->
                absorbCellArray(digitObj.optJSONArray("cells"))
                absorbCellArray(digitObj.optJSONArray("positions"))
                absorbCellArray(digitObj.optJSONArray("cell_refs"))
            }

            listOf("digit_locations", "candidates_by_digit", "digits_to_cells", "locations_by_digit").forEach { key ->
                val byDigit = obj.optJSONObject(key) ?: return@forEach
                absorbCellArray(byDigit.optJSONArray(requestedDigit.toString()))
                byDigit.optJSONObject(requestedDigit.toString())?.let { digitObj ->
                    absorbCellArray(digitObj.optJSONArray("cells"))
                    absorbCellArray(digitObj.optJSONArray("positions"))
                    absorbCellArray(digitObj.optJSONArray("cell_refs"))
                }
            }

            listOf("entries", "digits", "digit_entries").forEach { key ->
                val arr = obj.optJSONArray(key) ?: return@forEach
                for (i in 0 until arr.length()) {
                    val entry = arr.optJSONObject(i) ?: continue
                    val digit = entry.optInt("digit", -1)
                    if (digit != requestedDigit) continue
                    absorbCellArray(entry.optJSONArray("cells"))
                    absorbCellArray(entry.optJSONArray("positions"))
                    absorbCellArray(entry.optJSONArray("cell_refs"))
                }
            }
        }

        fun extractHouseDigitCellsFromHouseMapV1(
            packet: JSONObject,
            houseScope: String?,
            requestedDigit: Int?
        ): List<String> {
            if (houseScope.isNullOrBlank() || requestedDigit !in 1..9) return emptyList()
            val requested = requestedDigit ?: return emptyList()

            val root = packet.optJSONObject("result") ?: packet
            val out = linkedSetOf<String>()

            fun absorbHouseObject(obj: JSONObject?) {
                if (obj == null || obj.length() == 0) return

                val candidateHouse =
                    firstNonBlankStringV1(
                        obj.optString("house_scope", null),
                        obj.optString("house", null),
                        obj.optString("house_ref", null),
                        obj.optString("ref", null)
                    )

                if (!candidateHouse.isNullOrBlank() && !candidateHouse.equals(houseScope, ignoreCase = true)) {
                    return
                }

                absorbDigitCellsObjectV1(obj, requested, out)
            }

            absorbHouseObject(root)

            listOf("house_candidate_maps", "maps", "houses", "results").forEach { key ->
                val arr = root.optJSONArray(key) ?: return@forEach
                for (i in 0 until arr.length()) {
                    absorbHouseObject(arr.optJSONObject(i))
                }
            }

            return sortCellsByGridOrderV1(out)
        }

        fun buildHouseDigitSeatGeometryV1(
            houseScope: String?,
            askedDigit: Int?,
            houseMapPacket: JSONObject,
            cellsCandidatesPacket: JSONObject,
            targetCell: String?
        ): JSONObject {
            if (houseScope.isNullOrBlank() || askedDigit !in 1..9) return JSONObject()
            val asked = askedDigit ?: return JSONObject()

            val cellDigitsMap = extractCellDigitsMapFromCellsCandidatesV1(cellsCandidatesPacket)

            val openSeatsFromCells =
                sortCellsByGridOrderV1(
                    cellDigitsMap.keys.filter { houseContainsCellV1(houseScope, it) }
                )

            val survivingSeatsFromHouseMap =
                extractHouseDigitCellsFromHouseMapV1(
                    packet = houseMapPacket,
                    houseScope = houseScope,
                    requestedDigit = asked
                )

            val derivedSurvivingSeats =
                if (survivingSeatsFromHouseMap.isNotEmpty()) {
                    survivingSeatsFromHouseMap
                } else {
                    sortCellsByGridOrderV1(
                        openSeatsFromCells.filter { cell ->
                            cellDigitsMap[cell]?.contains(asked) == true
                        }
                    )
                }

            val openSeats =
                when {
                    openSeatsFromCells.isNotEmpty() -> openSeatsFromCells
                    derivedSurvivingSeats.isNotEmpty() -> derivedSurvivingSeats
                    else -> emptyList()
                }

            if (openSeats.isEmpty() && derivedSurvivingSeats.isEmpty()) return JSONObject()

            val seatStatusMap = JSONArray().apply {
                openSeats.forEach { cell ->
                    val survives = derivedSurvivingSeats.contains(cell)
                    put(
                        JSONObject().apply {
                            put("cell", cell)
                            put("status", if (survives) "SURVIVES" else "BLOCKED")
                            put("candidate_digits", JSONArray(cellDigitsMap[cell] ?: emptyList<Int>()))
                            put("blocked_by", JSONArray())
                            if (survives) {
                                put("survival_reason", "DIGIT_REMAINS_LEGAL_IN_HOUSE")
                            } else {
                                put("spoken_receipt", "This square is open in the house, but not for this digit.")
                            }
                        }
                    )
                }
            }

            val rivalSeatEliminations = JSONArray().apply {
                for (i in 0 until seatStatusMap.length()) {
                    val row = seatStatusMap.optJSONObject(i) ?: continue
                    if (row.optString("status", "") == "BLOCKED") put(row)
                }
            }

            val onlyPlaceOutcome =
                if (derivedSurvivingSeats.size == 1) {
                    JSONObject().apply {
                        put("digit", asked)
                        put("cell", derivedSurvivingSeats.first())
                        put("house_scope", houseScope)
                    }
                } else {
                    JSONObject()
                }

            val survivorSummary =
                if (derivedSurvivingSeats.size == 1) {
                    JSONObject().apply {
                        put("surviving_digit", asked)
                        put("winning_cell", derivedSurvivingSeats.first())
                        put("only_place_in_house", true)
                        put("surviving_seats", JSONArray(derivedSurvivingSeats))
                    }
                } else {
                    JSONObject()
                }

            return JSONObject().apply {
                put("geometry_kind", "HOUSE_DIGIT_SEAT_MAP")
                put("house_scope", houseScope)
                put("asked_digit", asked)
                put("open_seats", JSONArray(openSeats))
                put("seat_status_map", seatStatusMap)
                put("ordered_rival_seat_eliminations", rivalSeatEliminations)
                put("surviving_seats", JSONArray(derivedSurvivingSeats))
                put("only_place_outcome", if (onlyPlaceOutcome.length() > 0) onlyPlaceOutcome else JSONObject.NULL)
                put(
                    "primary_spotlight",
                    when {
                        derivedSurvivingSeats.size == 1 -> derivedSurvivingSeats.first()
                        !targetCell.isNullOrBlank() -> targetCell
                        openSeats.isNotEmpty() -> openSeats.first()
                        else -> JSONObject.NULL
                    }
                )
                put("house_survivor_summary", survivorSummary)
            }
        }

        fun parseDigitFromAnyFieldV1(obj: JSONObject?): Int? {
            if (obj == null) return null
            listOf("digit", "blocked_digit", "candidate_digit", "asked_digit", "target_digit").forEach { key ->
                val d = obj.optInt(key, -1)
                if (d in 1..9) return d
            }
            return null
        }

        fun parseCellFromAnyFieldV1(obj: JSONObject?): String? {
            if (obj == null) return null
            return firstNonBlankStringV1(
                obj.optString("cell", null),
                obj.optString("cell_ref", null),
                obj.optString("blocker_cell", null),
                obj.optString("blocker_cell_ref", null),
                obj.optString("ref", null)
            )
        }

        fun parseHouseFromAnyFieldV1(obj: JSONObject?): String? {
            if (obj == null) return null
            return firstNonBlankStringV1(
                obj.optString("house_scope", null),
                obj.optString("house", null),
                obj.optString("house_ref", null),
                obj.optString("blocker_house", null),
                obj.optString("blocking_house", null)
            )
        }

        fun parseRelationFromAnyFieldV1(obj: JSONObject?): String? {
            if (obj == null) return null
            return firstNonBlankStringV1(
                obj.optString("relation", null),
                obj.optString("blocking_relation", null)
            )
        }

        fun toIntSetFromJsonArrayV1(arr: JSONArray?): MutableSet<Int> {
            val out = linkedSetOf<Int>()
            if (arr == null) return out
            for (i in 0 until arr.length()) {
                val d = arr.optInt(i, -1)
                if (d in 1..9) out += d
            }
            return out
        }

        fun canonicalThreeHouseRefsV1(
            targetCell: String?,
            houses: JSONArray?
        ): List<String> {
            val ordered = linkedSetOf<String>()

            parseCellRefV1(targetCell)?.let { (row, col) ->
                ordered += "row$row"
                ordered += "col$col"
                val box = ((row - 1) / 3) * 3 + ((col - 1) / 3) + 1
                ordered += "box$box"
            }

            if (houses != null) {
                for (i in 0 until houses.length()) {
                    val house = firstNonBlankStringV1(houses.optString(i, null))
                    if (!house.isNullOrBlank()) ordered += house
                }
            }

            return ordered.toList()
        }

        fun relationForHouseV1(houseRef: String?): String? =
            when {
                houseRef.isNullOrBlank() -> null
                houseRef.startsWith("row", ignoreCase = true) -> "SAME_ROW"
                houseRef.startsWith("col", ignoreCase = true) -> "SAME_COL"
                houseRef.startsWith("box", ignoreCase = true) -> "SAME_BOX"
                else -> null
            }

        fun inferSharedHouseForReceiptV1(
            targetCell: String,
            blockerCell: String?,
            preferredHouse: String?,
            canonicalHouses: List<String>
        ): String? {
            if (!preferredHouse.isNullOrBlank()) return preferredHouse
            if (blockerCell.isNullOrBlank()) return null
            return canonicalHouses.firstOrNull { houseContainsCellV1(it, blockerCell) }
        }

        fun addBlockerReceiptV1(
            receiptsByHouse: LinkedHashMap<String, MutableList<JSONObject>>,
            allReceipts: MutableList<JSONObject>,
            seenKeys: MutableSet<String>,
            targetCell: String,
            digit: Int?,
            houseRef: String?,
            blockerCell: String?,
            relation: String?,
            source: String
        ) {
            if (digit !in 1..9) return

            val normalizedHouse = firstNonBlankStringV1(houseRef)
            val normalizedCell = firstNonBlankStringV1(blockerCell)
            val normalizedRelation = firstNonBlankStringV1(relation) ?: relationForHouseV1(normalizedHouse)
            val dedupeKey = listOf(digit, normalizedHouse ?: "", normalizedCell ?: "", normalizedRelation ?: "").joinToString("|")
            if (!seenKeys.add(dedupeKey)) return

            val receipt = JSONObject().apply {
                put("digit", digit)
                put("blocking_house", normalizedHouse ?: JSONObject.NULL)
                put("house_scope", normalizedHouse ?: JSONObject.NULL)
                put("blocker_cell", normalizedCell ?: JSONObject.NULL)
                put("relation", normalizedRelation ?: JSONObject.NULL)
                put("target_cell", targetCell)
                put("source", source)
                put(
                    "spoken_receipt",
                    when {
                        !normalizedHouse.isNullOrBlank() && !normalizedCell.isNullOrBlank() ->
                            "Digit $digit is blocked from $targetCell by $normalizedCell in $normalizedHouse."
                        !normalizedHouse.isNullOrBlank() ->
                            "Digit $digit is blocked from $targetCell by $normalizedHouse."
                        !normalizedCell.isNullOrBlank() ->
                            "Digit $digit is blocked from $targetCell by $normalizedCell."
                        else ->
                            "Digit $digit is blocked from $targetCell."
                    }
                )
            }

            allReceipts += receipt
            if (!normalizedHouse.isNullOrBlank()) {
                receiptsByHouse.getOrPut(normalizedHouse) { mutableListOf() }.add(receipt)
            }
        }

        fun absorbBlockedDigitsByHouseFromGeometryV1(
            geometry: JSONObject?,
            targetCell: String,
            canonicalHouses: List<String>,
            receiptsByHouse: LinkedHashMap<String, MutableList<JSONObject>>,
            allReceipts: MutableList<JSONObject>,
            seenKeys: MutableSet<String>,
            source: String
        ) {
            val byHouse = geometry?.optJSONObject("blocked_digits_by_house") ?: return
            val keys = byHouse.keys()
            while (keys.hasNext()) {
                val houseKey = keys.next()
                val normalizedHouse =
                    inferSharedHouseForReceiptV1(
                        targetCell = targetCell,
                        blockerCell = null,
                        preferredHouse = houseKey,
                        canonicalHouses = canonicalHouses
                    )

                val arr = byHouse.optJSONArray(houseKey) ?: continue
                for (i in 0 until arr.length()) {
                    val obj = arr.optJSONObject(i)
                    if (obj != null) {
                        val digit = parseDigitFromAnyFieldV1(obj)
                        val blockerCell = parseCellFromAnyFieldV1(obj)
                        val house =
                            inferSharedHouseForReceiptV1(
                                targetCell = targetCell,
                                blockerCell = blockerCell,
                                preferredHouse = parseHouseFromAnyFieldV1(obj) ?: normalizedHouse,
                                canonicalHouses = canonicalHouses
                            )
                        val relation = parseRelationFromAnyFieldV1(obj)
                        addBlockerReceiptV1(
                            receiptsByHouse = receiptsByHouse,
                            allReceipts = allReceipts,
                            seenKeys = seenKeys,
                            targetCell = targetCell,
                            digit = digit,
                            houseRef = house,
                            blockerCell = blockerCell,
                            relation = relation,
                            source = source
                        )
                    } else {
                        val digit = arr.optInt(i, -1)
                        addBlockerReceiptV1(
                            receiptsByHouse = receiptsByHouse,
                            allReceipts = allReceipts,
                            seenKeys = seenKeys,
                            targetCell = targetCell,
                            digit = if (digit in 1..9) digit else null,
                            houseRef = normalizedHouse,
                            blockerCell = null,
                            relation = relationForHouseV1(normalizedHouse),
                            source = source
                        )
                    }
                }
            }
        }

        fun absorbBlockerReceiptsArrayV1(
            arr: JSONArray?,
            targetCell: String,
            canonicalHouses: List<String>,
            receiptsByHouse: LinkedHashMap<String, MutableList<JSONObject>>,
            allReceipts: MutableList<JSONObject>,
            seenKeys: MutableSet<String>,
            source: String
        ) {
            if (arr == null) return
            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                val digit = parseDigitFromAnyFieldV1(obj)
                val blockerCell = parseCellFromAnyFieldV1(obj)
                val house =
                    inferSharedHouseForReceiptV1(
                        targetCell = targetCell,
                        blockerCell = blockerCell,
                        preferredHouse = parseHouseFromAnyFieldV1(obj),
                        canonicalHouses = canonicalHouses
                    )
                val relation = parseRelationFromAnyFieldV1(obj)
                addBlockerReceiptV1(
                    receiptsByHouse = receiptsByHouse,
                    allReceipts = allReceipts,
                    seenKeys = seenKeys,
                    targetCell = targetCell,
                    digit = digit,
                    houseRef = house,
                    blockerCell = blockerCell,
                    relation = relation,
                    source = source
                )
            }
        }

        fun absorbBlockerRowsPacketV1(
            blockersPacket: JSONObject,
            targetCell: String,
            canonicalHouses: List<String>,
            receiptsByHouse: LinkedHashMap<String, MutableList<JSONObject>>,
            allReceipts: MutableList<JSONObject>,
            seenKeys: MutableSet<String>
        ) {
            val blockerRows =
                blockersPacket.optJSONArray("blocker_rows")
                    ?: blockersPacket.optJSONArray("witness_rows")
                    ?: JSONArray()

            for (i in 0 until blockerRows.length()) {
                val row = blockerRows.optJSONObject(i) ?: continue
                val digit = parseDigitFromAnyFieldV1(row)
                val blockerCell = parseCellFromAnyFieldV1(row)
                val house =
                    inferSharedHouseForReceiptV1(
                        targetCell = targetCell,
                        blockerCell = blockerCell,
                        preferredHouse = parseHouseFromAnyFieldV1(row),
                        canonicalHouses = canonicalHouses
                    )
                val relation = parseRelationFromAnyFieldV1(row)
                addBlockerReceiptV1(
                    receiptsByHouse = receiptsByHouse,
                    allReceipts = allReceipts,
                    seenKeys = seenKeys,
                    targetCell = targetCell,
                    digit = digit,
                    houseRef = house,
                    blockerCell = blockerCell,
                    relation = relation,
                    source = "solver_blockers_packet"
                )
            }
        }

        fun sortedReceiptArrayV1(receipts: List<JSONObject>): JSONArray =
            JSONArray().apply {
                receipts
                    .sortedWith(
                        compareBy<JSONObject>(
                            { it.optInt("digit", 99) },
                            { it.optString("blocking_house", "~") },
                            { it.optString("blocker_cell", "~") }
                        )
                    )
                    .forEach { put(it) }
            }

        fun enrichCellThreeHouseGeometryFromBlockersV1(
            seedGeometry: JSONObject,
            narrativeGeometry: JSONObject?,
            targetCell: String?,
            askedDigit: Int?,
            houses: JSONArray?,
            targetBeforeState: JSONObject?,
            blockersPacket: JSONObject
        ): JSONObject {
            if (targetCell.isNullOrBlank()) return seedGeometry

            val seedKind =
                firstNonBlankStringV1(seedGeometry.optString("geometry_kind", null))
                    ?.trim()
                    ?.uppercase()
            val narrativeKind =
                firstNonBlankStringV1(narrativeGeometry?.optString("geometry_kind", null))
                    ?.trim()
                    ?.uppercase()

            if (
                (seedKind != null && seedKind != "CELL_THREE_HOUSE_UNIVERSE") ||
                (narrativeKind != null && narrativeKind != "CELL_THREE_HOUSE_UNIVERSE")
            ) {
                return seedGeometry
            }

            val canonicalHouses = canonicalThreeHouseRefsV1(targetCell, houses)

            val targetDigits =
                toIntSetFromJsonArrayV1(
                    targetBeforeState?.optJSONArray("digits")
                        ?: seedGeometry.optJSONArray("surviving_digits")
                        ?: narrativeGeometry?.optJSONArray("surviving_digits")
                )

            val receiptsByHouse = linkedMapOf<String, MutableList<JSONObject>>()
            val allReceipts = mutableListOf<JSONObject>()
            val seenKeys = linkedSetOf<String>()

            absorbBlockedDigitsByHouseFromGeometryV1(
                geometry = narrativeGeometry,
                targetCell = targetCell,
                canonicalHouses = canonicalHouses,
                receiptsByHouse = receiptsByHouse,
                allReceipts = allReceipts,
                seenKeys = seenKeys,
                source = "narrative_geometry.blocked_digits_by_house"
            )

            absorbBlockedDigitsByHouseFromGeometryV1(
                geometry = seedGeometry,
                targetCell = targetCell,
                canonicalHouses = canonicalHouses,
                receiptsByHouse = receiptsByHouse,
                allReceipts = allReceipts,
                seenKeys = seenKeys,
                source = "seed_geometry.blocked_digits_by_house"
            )

            absorbBlockerReceiptsArrayV1(
                arr = narrativeGeometry?.optJSONArray("blocker_receipts"),
                targetCell = targetCell,
                canonicalHouses = canonicalHouses,
                receiptsByHouse = receiptsByHouse,
                allReceipts = allReceipts,
                seenKeys = seenKeys,
                source = "narrative_geometry.blocker_receipts"
            )

            absorbBlockerReceiptsArrayV1(
                arr = seedGeometry.optJSONArray("blocker_receipts"),
                targetCell = targetCell,
                canonicalHouses = canonicalHouses,
                receiptsByHouse = receiptsByHouse,
                allReceipts = allReceipts,
                seenKeys = seenKeys,
                source = "seed_geometry.blocker_receipts"
            )

            absorbBlockerRowsPacketV1(
                blockersPacket = blockersPacket,
                targetCell = targetCell,
                canonicalHouses = canonicalHouses,
                receiptsByHouse = receiptsByHouse,
                allReceipts = allReceipts,
                seenKeys = seenKeys
            )

            val survivingDigits =
                if (targetDigits.isNotEmpty()) {
                    targetDigits.toList().sorted()
                } else {
                    emptyList()
                }

            val mergedBlockedDigits = linkedSetOf<Int>()
            if (survivingDigits.isNotEmpty()) {
                for (d in 1..9) {
                    if (!survivingDigits.contains(d)) mergedBlockedDigits += d
                }
            }
            allReceipts.forEach { receipt ->
                val digit = receipt.optInt("digit", -1)
                if (digit in 1..9) mergedBlockedDigits += digit
            }

            val blockerReceiptsJson = sortedReceiptArrayV1(allReceipts)

            val blockedDigitsByHouseJson = JSONObject().apply {
                canonicalHouses.forEach { house ->
                    put(house, sortedReceiptArrayV1(receiptsByHouse[house].orEmpty()))
                }
                receiptsByHouse.keys
                    .filterNot { canonicalHouses.contains(it) }
                    .sorted()
                    .forEach { house ->
                        put(house, sortedReceiptArrayV1(receiptsByHouse[house].orEmpty()))
                    }
            }

            val candidateStatusMap = JSONArray().apply {
                for (digit in 1..9) {
                    val survives = survivingDigits.contains(digit)
                    val matchingReceipts = JSONArray().apply {
                        allReceipts
                            .filter { it.optInt("digit", -1) == digit }
                            .sortedWith(
                                compareBy<JSONObject>(
                                    { it.optString("blocking_house", "~") },
                                    { it.optString("blocker_cell", "~") }
                                )
                            )
                            .forEach { put(it) }
                    }
                    put(
                        JSONObject().apply {
                            put("digit", digit)
                            put("status", if (survives) "SURVIVES" else "BLOCKED")
                            put("receipts", matchingReceipts)
                        }
                    )
                }
            }

            val scanOrder = JSONArray().apply {
                put("ROW")
                put("COLUMN")
                put("BOX")
            }

            val housesJson = JSONArray().apply {
                canonicalHouses.forEach { put(it) }
            }

            return JSONObject(seedGeometry.toString()).apply {
                put("geometry_kind", "CELL_THREE_HOUSE_UNIVERSE")
                put("target_cell", targetCell)
                put("asked_digit", askedDigit ?: opt("asked_digit") ?: JSONObject.NULL)
                put("houses", housesJson)
                put("blocked_digits_by_house", blockedDigitsByHouseJson)
                put("merged_blocked_digits", JSONArray(mergedBlockedDigits.toList().sorted()))
                put("surviving_digits", JSONArray(survivingDigits))
                put("candidate_status_map", candidateStatusMap)
                put("blocker_receipts", blockerReceiptsJson)
                put("scan_order", scanOrder)
                put("primary_spotlight", targetCell)
            }
        }

        val normalized = projectNormalizedDetourMoveProof(replyRequest)
        if (normalized.length() > 0) {
            val question = normalized.optJSONObject("question") ?: JSONObject()
            val proofTruth = normalized.optJSONObject("proof_truth") ?: JSONObject()
            val routeContext = normalized.optJSONObject("route_context") ?: JSONObject()
            val scope = normalized.optJSONObject("scope") ?: JSONObject()
            val answerTruthSrc = normalized.optJSONObject("answer_truth") ?: JSONObject()
            val proofOutcomeSrc = proofTruth.optJSONObject("proof_outcome") ?: JSONObject()

            val proofLadderSrc = proofTruth.optJSONObject("proof_ladder") ?: JSONObject()
            val overlayContext = normalized.optJSONObject("overlay_context") ?: JSONObject()
            val supportMeta = normalized.optJSONObject("support") ?: JSONObject()
            val debugMeta = normalized.optJSONObject("debug") ?: JSONObject()

            val narrativeContext = projectDetourNarrativeContext(replyRequest)
            val dominantAtom = narrativeContext.optJSONObject("dominant_atom") ?: JSONObject()
            val dominantAtomLocalProofGeometry =
                dominantAtom.optJSONObject("local_proof_geometry")
                    ?: narrativeContext.optJSONObject("local_proof_geometry")
                    ?: JSONObject()


            val targetCell = firstNonBlankStringV1(
                question.optString("target_cell", null),
                scope.optJSONArray("cells")?.optString(0, null)
            )

            val askedDigit = run {
                val q = question.optInt("asked_digit", -1)
                if (q in 1..9) return@run q
                val t = question.optInt("target_digit", -1)
                if (t in 1..9) return@run t
                null
            }

            val challengeLane = firstNonBlankStringV1(
                normalized.optString("challenge_lane", null),
                proofTruth.optString("challenge_lane", null)
            ) ?: "ELIMINATION_LEGITIMACY"

            val proofObject = firstNonBlankStringV1(
                normalized.optString("proof_object", null),
                proofTruth.optString("proof_object", null),
                proofTruth.optString("claim_kind", null)
            ) ?: "LOCAL_PROOF_INSUFFICIENT"

            val methodFamily = firstNonBlankStringV1(
                normalized.optString("method_family", null),
                proofTruth.optString("method_family", null)
            ) ?: "PERSISTENCE_OR_INSUFFICIENCY"

            val canonicalMethodFamily = firstNonBlankStringV1(
                normalized.optString("canonical_method_family", null),
                proofTruth.optString("canonical_method_family", null)
            ) ?: methodFamily

            val answerPolarity = firstNonBlankStringV1(
                answerTruthSrc.optString("answer_polarity", null),
                proofTruth.optString("answer_polarity", null)
            ) ?: "NOT_LOCALLY_PROVED"

            val localTruthStatus = firstNonBlankStringV1(
                answerTruthSrc.optString("local_truth_status", null),
                proofTruth.optString("local_truth_status", null)
            ) ?: "NOT_LOCALLY_ESTABLISHED"

            val nonproofReason = firstNonBlankStringV1(
                proofOutcomeSrc.optString("nonproof_reason", null)
            )

            val archetype = firstNonBlankStringV1(
                normalized.optString("narrative_archetype", null),
                proofTruth.optString("narrative_archetype", null)
            ) ?: archetypeForProof(
                methodFamily = methodFamily,
                proofObject = proofObject,
                challengeLane = challengeLane,
                answerPolarity = answerPolarity,
                nonproofReason = nonproofReason,
                rivalCell = firstNonBlankStringV1(question.optString("rival_cell", null)),
                claimedTechniqueId = firstNonBlankStringV1(question.optString("claimed_technique_id", null))
            )

            val doctrineId = firstNonBlankStringV1(
                normalized.optString("doctrine_id", null),
                proofTruth.optString("doctrine_id", null)
            ) ?: doctrineForArchetype(archetype)

            val actorModelId = firstNonBlankStringV1(
                normalized.optString("actor_model", null),
                proofTruth.optString("actor_model", null)
            ) ?: "LOCAL_SINGLE_SCOPE"

            val speechSkeleton =
                normalized.optJSONArray("speech_skeleton")
                    ?: proofTruth.optJSONArray("speech_skeleton")
                    ?: JSONArray()

            val targetBeforeState = proofTruth.optJSONObject("target_before_state") ?: JSONObject()
            val targetAfterState = proofTruth.optJSONObject("target_after_state") ?: JSONObject()
            val survivorSummary = proofTruth.optJSONObject("survivor_summary") ?: JSONObject()
            val contrastSummary = proofTruth.optJSONObject("contrast_summary") ?: JSONObject()
            val techniqueLegitimacy = proofTruth.optJSONObject("technique_legitimacy") ?: JSONObject()

            val questionFrame = JSONObject().apply {
                put("user_ask_kind", firstNonBlankStringV1(question.optString("user_ask_kind", null)) ?: JSONObject.NULL)
                put("challenge_lane", challengeLane)
                put("proof_object", proofObject)
                put("asked_cell", targetCell ?: JSONObject.NULL)
                put("asked_digit", askedDigit ?: JSONObject.NULL)
                put("asked_house", firstNonBlankStringV1(
                    proofTruth.optString("house_scope", null),
                    scope.optJSONArray("houses")?.optString(0, null)
                ) ?: JSONObject.NULL)
                put("contrast_cell", firstNonBlankStringV1(question.optString("rival_cell", null)) ?: JSONObject.NULL)
                put("contrast_digit", question.opt("contrast_digit") ?: JSONObject.NULL)
                put("claimed_technique_id", firstNonBlankStringV1(question.optString("claimed_technique_id", null)) ?: JSONObject.NULL)
                put(
                    "user_question_text",
                    firstNonBlankStringV1(
                        question.optString("central_question", null),
                        answerTruthSrc.optString("short_answer", null)
                    ) ?: JSONObject.NULL
                )
            }

            val answerTruth = JSONObject().apply {
                put("answer_polarity", answerPolarity)
                put(
                    "short_answer",
                    firstNonBlankStringV1(
                        answerTruthSrc.optString("short_answer", null),
                        proofTruth.optString("proof_claim", null),
                        proofTruth.optString("decisive_fact", null)
                    ) ?: JSONObject.NULL
                )
                put(
                    "one_sentence_claim",
                    firstNonBlankStringV1(
                        answerTruthSrc.optString("one_sentence_claim", null),
                        proofTruth.optString("decisive_fact", null),
                        proofTruth.optString("proof_claim", null)
                    ) ?: JSONObject.NULL
                )
                put("local_truth_status", localTruthStatus)
            }

            val proofObjectJson = JSONObject().apply {
                put("proof_object", proofObject)
                put("claim_kind", firstNonBlankStringV1(proofTruth.optString("claim_kind", null)) ?: JSONObject.NULL)
                put("challenge_lane", challengeLane)
                put("elimination_kind", firstNonBlankStringV1(proofTruth.optString("elimination_kind", null)) ?: JSONObject.NULL)
            }

            val proofMethod = JSONObject().apply {
                put("method_family", methodFamily)
                put("canonical_method_family", canonicalMethodFamily)
                put("proof_scope", proofScopeFor(methodFamily, nonproofReason))
                put("proof_strength", proofStrengthFor(methodFamily, answerPolarity))
            }

            val narrativeArchetype = JSONObject().apply {
                put("id", archetype)
                put("enum_name", packetArchetypeEnumV1(archetype).name)
                put("family", when (archetype.trim().uppercase()) {
                    "LOCAL_CONTRADICTION_SPOTLIGHT" -> "LOCAL_CONTRADICTION"
                    "SURVIVOR_LADDER" -> "LOCAL_SURVIVOR"
                    "CONTRAST_DUEL" -> "LOCAL_CONTRAST"
                    "PATTERN_LEGITIMACY_CHECK" -> "PATTERN_LEGITIMACY"
                    "HONEST_INSUFFICIENCY_ANSWER" -> "HONEST_INSUFFICIENCY"
                    else -> "LOCAL_PROOF"
                })
            }

            val doctrine = JSONObject().apply {
                put("id", doctrineId)
                put("answer_shape", when (archetype.trim().uppercase()) {
                    "LOCAL_CONTRADICTION_SPOTLIGHT" -> "LOCAL_BLOCKER_EXPLANATION"
                    "SURVIVOR_LADDER" -> "SURVIVOR_PROOF"
                    "CONTRAST_DUEL" -> "RIVAL_COMPARISON"
                    "PATTERN_LEGITIMACY_CHECK" -> "PATTERN_VALIDATION"
                    else -> "HONEST_LIMITED_ANSWER"
                })
                put("must_answer_local_question_first", true)
                put("must_not_reopen_route", true)
                put("must_not_board_audit", true)
                put("must_not_commit", true)
                put("must_end_with_handback", false)
            }

            val actorModel = JSONObject().apply {
                put("id", actorModelId)
                put("supports_primary_target", true)
                put("supports_rival_frame", contrastSummary.length() > 0)
                put("supports_pattern_frame", techniqueLegitimacy.length() > 0)
                put("supports_house_frame", firstNonBlankStringV1(proofTruth.optString("house_scope", null)) != null)
            }


            val storyFocus = JSONObject().apply {
                put("scope", firstNonBlankStringV1(scope.optString("kind", null), scope.optString("ref", null)) ?: JSONObject.NULL)
                put("primary_cell", targetCell ?: JSONObject.NULL)
                put("asked_digit", askedDigit ?: JSONObject.NULL)
                put(
                    "house_scope",
                    firstNonBlankStringV1(
                        proofTruth.optString("house_scope", null),
                        scope.optJSONArray("houses")?.optString(0, null)
                    ) ?: JSONObject.NULL
                )
                put("focus_cells", scope.optJSONArray("cells") ?: JSONArray())
                put("focus_houses", scope.optJSONArray("houses") ?: JSONArray())
                put("overlay_story_kind", firstNonBlankStringV1(overlayContext.optString("overlay_story_kind", null)) ?: JSONObject.NULL)
            }

            val storyQuestion = JSONObject().apply {
                put("user_ask_kind", firstNonBlankStringV1(question.optString("user_ask_kind", null)) ?: JSONObject.NULL)
                put(
                    "central_question",
                    firstNonBlankStringV1(
                        question.optString("central_question", null),
                        questionFrame.optString("user_question_text", null),
                        answerTruth.optString("short_answer", null)
                    ) ?: JSONObject.NULL
                )
                put("challenge_lane", challengeLane)
                put("proof_object", proofObject)
                put("target_relation", when (archetype.trim().uppercase()) {
                    "LOCAL_CONTRADICTION_SPOTLIGHT" -> "BLOCK_OR_RULE_OUT"
                    "SURVIVOR_LADDER" -> "SURVIVE_OR_ONLY_PLACE"
                    "CONTRAST_DUEL" -> "COMPARE_RIVALS"
                    "PATTERN_LEGITIMACY_CHECK" -> "VALIDATE_PATTERN"
                    else -> "ESTABLISH_LOCAL_LIMIT"
                })
            }

            val storyActors = JSONObject().apply {
                put("target_cell", targetCell ?: JSONObject.NULL)
                put("asked_digit", askedDigit ?: JSONObject.NULL)
                put("rival_cell", firstNonBlankStringV1(question.optString("rival_cell", null)) ?: JSONObject.NULL)
                put("rival_digit", question.opt("contrast_digit") ?: JSONObject.NULL)
                put("blocker_house", firstNonBlankStringV1(proofTruth.optString("house_scope", null)) ?: JSONObject.NULL)
                put("survivor_summary", survivorSummary)
                put("contrast_summary", contrastSummary)
                put("technique_legitimacy", techniqueLegitimacy)
            }

            val solverCellsCandidatesPacket = projectSolverCellsCandidatesPacket(replyRequest)
            val solverHouseMapPacket = projectSolverHouseCandidateMapPacket(replyRequest)

            val solverBlockersPacket = projectSolverCellDigitBlockersPacket(replyRequest)

            val packetLocalProofGeometrySeed =
                proofTruth.optJSONObject("local_proof_geometry")
                    ?: normalized.optJSONObject("local_proof_geometry")
                    ?: JSONObject()

            val packetLocalProofGeometry =
                enrichCellThreeHouseGeometryFromBlockersV1(
                    seedGeometry = packetLocalProofGeometrySeed,
                    narrativeGeometry = dominantAtomLocalProofGeometry,
                    targetCell = targetCell,
                    askedDigit = askedDigit,
                    houses = scope.optJSONArray("houses"),
                    targetBeforeState = targetBeforeState,
                    blockersPacket = solverBlockersPacket
                )

            val requestedHouseScope =
                firstNonBlankStringV1(
                    proofTruth.optString("house_scope", null),
                    scope.optJSONArray("houses")?.optString(0, null)
                )

            val shouldPreferHouseSeatGeometry =
                !requestedHouseScope.isNullOrBlank() && (
                        challengeLane.trim().uppercase() == "FORCEDNESS_OR_UNIQUENESS" ||
                                methodFamily.trim().uppercase() == "HOUSE_UNIQUENESS" ||
                                methodFamily.trim().uppercase() == "RIVAL_ELIMINATION_LADDER" ||
                                proofObject.trim().uppercase() == "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE" ||
                                proofObject.trim().uppercase() == "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL"
                        )

            val fallbackHouseSeatGeometry =
                buildHouseDigitSeatGeometryV1(
                    houseScope = requestedHouseScope,
                    askedDigit = askedDigit,
                    houseMapPacket = solverHouseMapPacket,
                    cellsCandidatesPacket = solverCellsCandidatesPacket,
                    targetCell = targetCell
                )

            val localProofGeometry =
                when {
                    shouldPreferHouseSeatGeometry &&
                            fallbackHouseSeatGeometry.length() > 0 &&
                            (
                                    packetLocalProofGeometry.length() == 0 ||
                                            firstNonBlankStringV1(packetLocalProofGeometry.optString("geometry_kind", null)).isNullOrBlank() ||
                                            firstNonBlankStringV1(packetLocalProofGeometry.optString("geometry_kind", null)) == "NONE"
                                    ) ->
                        fallbackHouseSeatGeometry

                    else ->
                        packetLocalProofGeometry
                }



            val proofLadderRows =
                proofLadderSrc.optJSONArray("rows")
                    ?: proofTruth.optJSONArray("bounded_proof_rows")
                    ?: JSONArray()

            val proofLadder = JSONObject().apply {
                put("rows", proofLadderRows)
                put("row_count", proofLadderRows.length())
                put("has_rows", proofLadderRows.length() > 0)
            }

            val proofOutcome = JSONObject().apply {
                put("surviving_digit", proofOutcomeSrc.opt("surviving_digit") ?: survivorSummary.opt("surviving_digit") ?: JSONObject.NULL)
                put("remaining_candidates", proofOutcomeSrc.optJSONArray("remaining_candidates") ?: targetBeforeState.optJSONArray("digits") ?: JSONArray())
                put("only_place_in_house", proofOutcomeSrc.opt("only_place_in_house") ?: JSONObject.NULL)
                put("forced_cell_value", proofOutcomeSrc.opt("forced_cell_value") ?: JSONObject.NULL)
                put("winning_cell", proofOutcomeSrc.opt("winning_cell") ?: JSONObject.NULL)
                put("winning_digit", proofOutcomeSrc.opt("winning_digit") ?: JSONObject.NULL)
                put("nonproof_reason", nonproofReason ?: JSONObject.NULL)
                put("overlay_story_kind", firstNonBlankStringV1(proofOutcomeSrc.optString("overlay_story_kind", null)) ?: JSONObject.NULL)
                put("challenge_lane", challengeLane)
                put("survivor_summary", survivorSummary)
                put("contrast_summary", contrastSummary)
                put("technique_legitimacy", techniqueLegitimacy)
            }


            val storyArc = JSONObject().apply {
                put("opening_mode", when (archetype.trim().uppercase()) {
                    "LOCAL_CONTRADICTION_SPOTLIGHT" -> "SPOTLIGHT_TARGET"
                    "SURVIVOR_LADDER" -> "SHOW_LIVE_CONTENDERS"
                    "CONTRAST_DUEL" -> "FRAME_DUEL"
                    "PATTERN_LEGITIMACY_CHECK" -> "NAME_PATTERN"
                    else -> "SPOTLIGHT_LOCAL_QUESTION"
                })
                put("motion_mode", when (archetype.trim().uppercase()) {
                    "LOCAL_CONTRADICTION_SPOTLIGHT" -> "BLOCKER_TO_CONTRADICTION"
                    "SURVIVOR_LADDER" -> "RIVAL_FAILURES_TO_SURVIVOR"
                    "CONTRAST_DUEL" -> "COMPARE_UNDER_SHARED_STANDARD"
                    "PATTERN_LEGITIMACY_CHECK" -> "STRUCTURE_TO_CONSEQUENCE"
                    else -> "BOUNDED_HONEST_READOUT"
                })
                put(
                    "local_landing_line",
                    firstNonBlankStringV1(
                        proofTruth.optString("decisive_fact", null),
                        answerTruth.optString("one_sentence_claim", null),
                        answerTruth.optString("short_answer", null)
                    ) ?: JSONObject.NULL
                )
                put("proof_row_count", proofLadderSrc.optJSONArray("rows")?.length() ?: 0)
            }


            val microStagePlan = JSONObject().apply {
                put(
                    "micro_setup",
                    JSONObject().apply {
                        put("goal", "SPOTLIGHT_LOCAL_CHALLENGE")
                        put("opening_mode", storyArc.opt("opening_mode") ?: JSONObject.NULL)
                        put("focus_scope", storyFocus.opt("scope") ?: JSONObject.NULL)
                        put("must_name_local_question", true)
                    }
                )
                put(
                    "micro_confrontation",
                    JSONObject().apply {
                        put("goal", "WALK_LOCAL_PROOF")
                        put("motion_mode", storyArc.opt("motion_mode") ?: JSONObject.NULL)
                        put("proof_row_count", storyArc.optInt("proof_row_count", 0))
                        put("prefer_ordered_ladder_when_present", true)
                    }
                )
                put(
                    "micro_resolution",
                    JSONObject().apply {
                        put("goal", "LAND_LOCAL_RESULT")
                        put("landing_line", storyArc.opt("local_landing_line") ?: JSONObject.NULL)
                        put("must_not_commit_move", true)
                        put("must_close_boundedly", true)
                    }
                )
                put(
                    "compression_mode",
                    when {
                        (proofLadder.optJSONArray("rows")?.length() ?: 0) >= 3 -> "FULL_THREE_BEAT"
                        (proofLadder.optJSONArray("rows")?.length() ?: 0) > 0 -> "LIGHT_THREE_BEAT"
                        else -> "MINIMAL_THREE_BEAT"
                    }
                )
            }

            val closureContract = JSONObject().apply {
                put("closure_mode", "AUTHORED_NATURAL_CLOSURE")
                put("return_target_kind", "CURRENT_MOVE")
                put("return_style", "GENTLE_ROUTE_RETURN")
                put("may_offer_followup", true)
                put("followup_style", "ONE_BOUNDED_OPTION")
                put("may_offer_return_now", true)
                put("must_not_sound_procedural", true)
                put("must_not_emit_stock_handback", true)
                put("must_not_use_internal_route_jargon", true)
                put("must_land_local_result_before_return_offer", true)
            }

            val visualLanguage = JSONObject().apply {
                put("may_use_spotlight_language", true)
                put("may_use_scene_language", true)
                put("may_use_actor_language", true)
                put("may_use_survival_language", archetype.trim().uppercase() == "SURVIVOR_LADDER")
                put("may_use_duel_language", archetype.trim().uppercase() == "CONTRAST_DUEL")
                put("may_use_pattern_visibility_language", archetype.trim().uppercase() == "PATTERN_LEGITIMACY_CHECK")
                put("must_stay_grounded_in_packet_truth", true)
            }


            val speechBoundary = JSONObject().apply {
                put("must_answer_local_question_first", true)
                put("must_not_reopen_route", true)
                put("must_not_board_audit", true)
                put("must_not_commit", true)
                put("may_briefly_bridge_back", true)
                put("preferred_answer_length", "SHORT_TO_MEDIUM")
                put("end_with_handback", false)
            }

            val handbackContext = JSONObject().apply {
                put("paused_move_exists", true)
                put("handback_line", JSONObject.NULL)
                put("return_target_kind", "CURRENT_MOVE")
                put("return_target_label", JSONObject.NULL)
                put("handover_mode", firstNonBlankStringV1(routeContext.optString("handover_mode", null)) ?: "RETURN_TO_CURRENT_MOVE")
                put("may_offer_return_now", true)
                put("may_offer_one_followup", true)
                put("preferred_return_style", "GENTLE")
                put("preferred_followup_style", "BOUNDED")
            }

            val overlayPlan = JSONObject().apply {
                put("overlay_story_kind", firstNonBlankStringV1(overlayContext.optString("overlay_story_kind", null)) ?: JSONObject.NULL)
                put("overlay_mode", firstNonBlankStringV1(overlayContext.optString("overlay_mode", null)) ?: "AUGMENT")
                put("primary_focus_cells", overlayContext.optJSONArray("focus_cells") ?: JSONArray())
                put("primary_focus_houses", overlayContext.optJSONArray("focus_houses") ?: JSONArray())
                put("secondary_focus_cells", overlayContext.optJSONArray("secondary_focus_cells") ?: JSONArray())
                put("deemphasize_cells", overlayContext.optJSONArray("deemphasize_cells") ?: JSONArray())
                put("reason_for_focus", firstNonBlankStringV1(overlayContext.optString("reason_for_focus", null)) ?: JSONObject.NULL)
                put("highlight_roles", overlayContext.optJSONObject("highlight_roles") ?: JSONObject())
            }

            val supportingFacts = JSONObject().apply {
                put("focus_cells", scope.optJSONArray("cells") ?: JSONArray())
                put("focus_houses", localProofGeometry.optJSONArray("houses") ?: scope.optJSONArray("houses") ?: JSONArray())
                put("candidate_snapshot", targetBeforeState)
                put("target_before_state", targetBeforeState)
                put("target_after_state", targetAfterState)
                put("witness_rows", proofTruth.optJSONArray("witness_rows") ?: JSONArray())
                put("house_candidate_map_excerpt", JSONObject())
                put("technique_structure_excerpt", techniqueLegitimacy)
                put("candidate_status_map", localProofGeometry.optJSONArray("candidate_status_map") ?: JSONArray())
                put("blocked_digits_by_house", localProofGeometry.optJSONObject("blocked_digits_by_house") ?: JSONObject())
                put("blocker_receipts", localProofGeometry.optJSONArray("blocker_receipts") ?: JSONArray())
                put("merged_blocked_digits", localProofGeometry.optJSONArray("merged_blocked_digits") ?: JSONArray())
                put("surviving_digits", localProofGeometry.optJSONArray("surviving_digits") ?: JSONArray())
                put("scan_order", localProofGeometry.optJSONArray("scan_order") ?: JSONArray())
            }

            val debugSupport = JSONObject().apply {
                if (supportMeta.length() > 0) put("normalizer_support", supportMeta)
                if (debugMeta.length() > 0) put("normalizer_debug", debugMeta)
            }

            val packet = ProofChallengePacketV1(
                challengeLane = challengeLane,
                questionFrame = questionFrame,
                storyFocus = storyFocus,
                storyQuestion = storyQuestion,
                answerTruth = answerTruth,
                proofObject = proofObjectJson,
                proofMethod = proofMethod,
                narrativeArchetype = narrativeArchetype,
                doctrine = doctrine,
                speechSkeleton = speechSkeleton,
                actorModel = actorModel,
                storyActors = storyActors,
                proofLadder = proofLadder,
                proofOutcome = proofOutcome,
                storyArc = storyArc,
                microStagePlan = microStagePlan,
                localProofGeometry = localProofGeometry,
                speechBoundary = speechBoundary,
                closureContract = closureContract,
                handbackContext = handbackContext,
                overlayPlan = overlayPlan,
                visualLanguage = visualLanguage,
                supportingFacts = supportingFacts,
                debugSupport = debugSupport
            ).toJson()

            runCatching {
                com.contextionary.sudoku.telemetry.ConversationTelemetry.emitPolicyTrace(
                    tag = "DETOUR_PROOF_PACKET_V2_BUILT",
                    data = mapOf(
                        "challenge_lane" to challengeLane,
                        "proof_object" to proofObject,
                        "method_family" to methodFamily,
                        "canonical_method_family" to canonicalMethodFamily,
                        "narrative_archetype" to archetype,
                        "doctrine_id" to doctrineId,
                        "speech_skeleton_count" to speechSkeleton.length(),
                        "proof_ladder_rows" to proofLadderRows.length(),
                        "geometry_kind" to (firstNonBlankStringV1(localProofGeometry.optString("geometry_kind", null)) ?: "null"),
                        "overlay_story_kind" to (firstNonBlankStringV1(proofOutcomeSrc.optString("overlay_story_kind", null)) ?: "null")
                    )
                )
            }

            return packet
        }



        val detourQuestionClass =
            activeDetourQuestionClassFromReplyRequestV1(replyRequest)
                ?.trim()
                ?.uppercase()

        if (
            detourQuestionClass != "PROOF_CHALLENGE" &&
            detourQuestionClass != "TARGET_CELL_QUERY"
        ) return JSONObject()



        val proofChallenge = projectProofChallengePacket(replyRequest)
        val targetCellQuery = projectTargetCellQueryPacket(replyRequest)
        val blockers = projectSolverCellDigitBlockersPacket(replyRequest)
        val solverCellsCandidates = projectSolverCellsCandidatesPacket(replyRequest)
        val solverHouseMap = projectSolverHouseCandidateMapPacket(replyRequest)
        val scopedSupport = projectSolverScopedSupportPacket(replyRequest)

        if (
            proofChallenge.length() == 0 &&
            targetCellQuery.length() == 0 &&
            blockers.length() == 0 &&
            solverCellsCandidates.length() == 0 &&
            solverHouseMap.length() == 0 &&
            scopedSupport.length() == 0
        ) return JSONObject()

        val targetCell = firstNonBlankStringV1(
            targetCellQuery.optString("target_cell", null),
            targetCellQuery.optString("cell", null),
            proofChallenge.optString("target_cell", null),
            blockers.optString("cell", null)
        )

        val houseScope = firstNonBlankStringV1(
            targetCellQuery.optString("house_scope", null),
            proofChallenge.optString("house_scope", null),
            blockers.optString("house_scope", null)
        )

        val proofClaim = firstNonBlankStringV1(
            proofChallenge.optString("proof_claim", null),
            targetCellQuery.optString("question_summary", null),
            scopedSupport.optString("support_summary", null)
        )

        val targetDigit = run {
            val fromTarget = targetCellQuery.optInt("target_digit", -1)
            if (fromTarget in 1..9) return@run fromTarget
            val fromProof = proofChallenge.optInt("target_digit", -1)
            if (fromProof in 1..9) return@run fromProof
            null
        }

        val boundedProofRows =
            proofChallenge.optJSONArray("bounded_proof_rows")
                ?: proofChallenge.optJSONArray("proof_rows")
                ?: scopedSupport.optJSONArray("bounded_rows")
                ?: JSONArray()

        val witnessRows =
            blockers.optJSONArray("blocker_rows")
                ?: blockers.optJSONArray("witness_rows")
                ?: JSONArray()

        val baseSurvivorSummary =
            targetCellQuery.optJSONObject("survivor_summary")
                ?: proofChallenge.optJSONObject("survivor_summary")
                ?: JSONObject()

        val houseSeatGeometry =
            buildHouseDigitSeatGeometryV1(
                houseScope = houseScope,
                askedDigit = targetDigit,
                houseMapPacket = solverHouseMap,
                cellsCandidatesPacket = solverCellsCandidates,
                targetCell = targetCell
            )

        val houseSurvivorSummary =
            houseSeatGeometry.optJSONObject("house_survivor_summary")
                ?: JSONObject()

        val survivorSummary =
            if (baseSurvivorSummary.length() > 0) {
                baseSurvivorSummary
            } else {
                houseSurvivorSummary
            }

        val effectiveChallengeLane =
            when {
                houseSeatGeometry.length() > 0 -> "FORCEDNESS_OR_UNIQUENESS"
                detourQuestionClass == "PROOF_CHALLENGE" -> "CANDIDATE_IMPOSSIBILITY"
                else -> "FORCEDNESS_OR_UNIQUENESS"
            }

        val effectiveUserAskKind =
            when {
                houseSeatGeometry.length() > 0 && houseSurvivorSummary.length() > 0 ->
                    "WHY_ONLY_PLACE"
                houseSeatGeometry.length() > 0 ->
                    "WHERE_DIGIT_CAN_STILL_GO_IN_HOUSE"
                detourQuestionClass == "PROOF_CHALLENGE" ->
                    "WHY_NOT_DIGIT"
                else ->
                    "WHY_TARGET"
            }

        val effectiveProofObject =
            when {
                houseSeatGeometry.length() > 0 && houseSurvivorSummary.length() > 0 ->
                    "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE"
                houseSeatGeometry.length() > 0 ->
                    "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL"
                detourQuestionClass == "PROOF_CHALLENGE" ->
                    "CELL_CANNOT_BE_DIGIT"
                else ->
                    "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE"
            }

        val effectiveProofClaim =
            firstNonBlankStringV1(
                proofClaim,
                if (houseSeatGeometry.length() > 0 && houseSurvivorSummary.length() > 0) {
                    val winningCell = houseSurvivorSummary.optString("winning_cell", "").ifBlank { null }
                    if (winningCell != null && targetDigit != null && !houseScope.isNullOrBlank()) {
                        "Digit $targetDigit can only go in $winningCell within $houseScope."
                    } else {
                        null
                    }
                } else if (houseSeatGeometry.length() > 0 && targetDigit != null && !houseScope.isNullOrBlank()) {
                    "Here are the remaining places where digit $targetDigit can still go in $houseScope."
                } else {
                    null
                }
            )

        val questionFrame = JSONObject().apply {
            put("user_ask_kind", effectiveUserAskKind)
            put("proof_object", effectiveProofObject)
            put("asked_cell", targetCell ?: JSONObject.NULL)
            put("asked_digit", targetDigit ?: JSONObject.NULL)
            put("asked_house", houseScope ?: JSONObject.NULL)
            put("contrast_cell", JSONObject.NULL)
            put("contrast_digit", JSONObject.NULL)
            put("user_question_text", effectiveProofClaim ?: JSONObject.NULL)
        }

        val answerTruth = JSONObject().apply {
            put(
                "answer_polarity",
                when {
                    witnessRows.length() > 0 -> "RULED_OUT"
                    houseSurvivorSummary.length() > 0 -> "ONLY_PLACE"
                    houseSeatGeometry.length() > 0 -> "STILL_OPEN_IN_HOUSE"
                    else -> "NOT_LOCALLY_PROVED"
                }
            )
            put("short_answer", effectiveProofClaim ?: JSONObject.NULL)
            put("one_sentence_claim", effectiveProofClaim ?: JSONObject.NULL)
            put(
                "local_truth_status",
                when {
                    witnessRows.length() > 0 -> "DIRECTLY_PROVED"
                    houseSurvivorSummary.length() > 0 -> "PROVED_BY_HOUSE_UNIQUENESS"
                    houseSeatGeometry.length() > 0 -> "DIRECTLY_READABLE_IN_HOUSE"
                    else -> "NOT_LOCALLY_ESTABLISHED"
                }
            )
        }

        val proofMethod = JSONObject().apply {
            val methodFamily =
                when {
                    witnessRows.length() > 0 -> "DIRECT_CONTRADICTION"
                    houseSurvivorSummary.length() > 0 -> "HOUSE_UNIQUENESS"
                    houseSeatGeometry.length() > 0 -> "RIVAL_ELIMINATION_LADDER"
                    else -> "PERSISTENCE_OR_INSUFFICIENCY"
                }
            val archetype = archetypeForProof(
                methodFamily = methodFamily,
                proofObject = effectiveProofObject,
                challengeLane = effectiveChallengeLane,
                answerPolarity =
                    when {
                        witnessRows.length() > 0 -> "RULED_OUT"
                        houseSurvivorSummary.length() > 0 -> "ONLY_PLACE"
                        houseSeatGeometry.length() > 0 -> "STILL_OPEN_IN_HOUSE"
                        else -> "NOT_LOCALLY_PROVED"
                    },
                nonproofReason =
                    if (witnessRows.length() == 0 && survivorSummary.length() == 0 && houseSeatGeometry.length() == 0) {
                        "NO_DIRECT_LOCAL_BLOCKER_PRESENT"
                    } else {
                        null
                    },
                rivalCell = null,
                claimedTechniqueId = null
            )
            put("method_family", methodFamily)
            put("narrative_archetype", archetype)
            put("doctrine_id", doctrineForArchetype(archetype))
            put("proof_scope", if (houseSeatGeometry.length() > 0) "HOUSE_LOCAL" else "CELL_LOCAL")
            put(
                "proof_strength",
                when {
                    witnessRows.length() > 0 -> "DIRECT"
                    houseSeatGeometry.length() > 0 -> "STRUCTURED"
                    else -> "NEGATIVE_HONESTY"
                }
            )
        }

        val proofLadder = JSONObject().apply {
            put("rows", boundedProofRows)
        }

        val proofOutcome = JSONObject().apply {
            put("surviving_digit", survivorSummary.opt("surviving_digit") ?: JSONObject.NULL)
            put("remaining_candidates", targetCellQuery.optJSONObject("target_before_state")?.optJSONArray("digits") ?: JSONArray())
            put("only_place_in_house", if (survivorSummary.optBoolean("only_place_in_house", false)) houseScope ?: JSONObject.NULL else JSONObject.NULL)
            put("forced_cell_value", JSONObject.NULL)
            put("winning_cell", targetCell ?: JSONObject.NULL)
            put("winning_digit", targetDigit ?: JSONObject.NULL)
            put("nonproof_reason", if (witnessRows.length() == 0 && survivorSummary.length() == 0) "NO_DIRECT_LOCAL_BLOCKER_PRESENT" else JSONObject.NULL)
        }


        val storyFocus = JSONObject().apply {
            put("scope", if (houseScope != null) "HOUSE_LOCAL" else "CELL_LOCAL")
            put("primary_cell", targetCell ?: JSONObject.NULL)
            put("asked_digit", targetDigit ?: JSONObject.NULL)
            put("house_scope", houseScope ?: JSONObject.NULL)
            put("focus_cells", JSONArray(listOfNotNull(targetCell)))
            put("focus_houses", JSONArray(listOfNotNull(houseScope)))
            put("overlay_story_kind", JSONObject.NULL)
        }

        val storyQuestion = JSONObject().apply {
            put("user_ask_kind", effectiveUserAskKind)
            put("central_question", effectiveProofClaim ?: JSONObject.NULL)
            put("challenge_lane", effectiveChallengeLane)
            put("proof_object", effectiveProofObject)
            put(
                "target_relation",
                if (houseSeatGeometry.length() > 0) "SURVIVE_OR_ONLY_PLACE"
                else if (detourQuestionClass == "PROOF_CHALLENGE") "BLOCK_OR_RULE_OUT"
                else "SURVIVE_OR_ONLY_PLACE"
            )
        }

        val storyActors = JSONObject().apply {
            put("target_cell", targetCell ?: JSONObject.NULL)
            put("asked_digit", targetDigit ?: JSONObject.NULL)
            put("rival_cell", JSONObject.NULL)
            put("rival_digit", JSONObject.NULL)
            put("blocker_house", houseScope ?: JSONObject.NULL)
            put("survivor_summary", survivorSummary)
            put("contrast_summary", JSONObject())
            put("technique_legitimacy", JSONObject())
        }

        val storyArc = JSONObject().apply {
            put(
                "opening_mode",
                when {
                    witnessRows.length() > 0 -> "SPOTLIGHT_TARGET"
                    houseSeatGeometry.length() > 0 -> "SHOW_LIVE_CONTENDERS"
                    else -> "SPOTLIGHT_LOCAL_QUESTION"
                }
            )
            put(
                "motion_mode",
                when {
                    witnessRows.length() > 0 -> "BLOCKER_TO_CONTRADICTION"
                    houseSeatGeometry.length() > 0 -> "RIVAL_FAILURES_TO_SURVIVOR"
                    else -> "BOUNDED_HONEST_READOUT"
                }
            )
            put("local_landing_line", effectiveProofClaim ?: JSONObject.NULL)
            put("proof_row_count", boundedProofRows.length())
        }

        val microStagePlan = JSONObject().apply {
            put(
                "micro_setup",
                JSONObject().apply {
                    put("goal", "SPOTLIGHT_LOCAL_CHALLENGE")
                    put("opening_mode", storyArc.opt("opening_mode") ?: JSONObject.NULL)
                    put("focus_scope", storyFocus.opt("scope") ?: JSONObject.NULL)
                    put("must_name_local_question", true)
                }
            )
            put(
                "micro_confrontation",
                JSONObject().apply {
                    put("goal", "WALK_LOCAL_PROOF")
                    put("motion_mode", storyArc.opt("motion_mode") ?: JSONObject.NULL)
                    put("proof_row_count", storyArc.optInt("proof_row_count", 0))
                    put("prefer_ordered_ladder_when_present", true)
                }
            )
            put(
                "micro_resolution",
                JSONObject().apply {
                    put("goal", "LAND_LOCAL_RESULT")
                    put("landing_line", storyArc.opt("local_landing_line") ?: JSONObject.NULL)
                    put("must_not_commit_move", true)
                    put("must_close_boundedly", true)
                }
            )
            put(
                "compression_mode",
                when {
                    boundedProofRows.length() >= 3 -> "FULL_THREE_BEAT"
                    boundedProofRows.length() > 0 -> "LIGHT_THREE_BEAT"
                    else -> "MINIMAL_THREE_BEAT"
                }
            )
        }

        val closureContract = JSONObject().apply {
            put("closure_mode", "AUTHORED_NATURAL_CLOSURE")
            put("return_target_kind", "CURRENT_MOVE")
            put("return_style", "GENTLE_ROUTE_RETURN")
            put("may_offer_followup", true)
            put("followup_style", "ONE_BOUNDED_OPTION")
            put("may_offer_return_now", true)
            put("must_not_sound_procedural", true)
            put("must_not_emit_stock_handback", true)
            put("must_not_use_internal_route_jargon", true)
            put("must_land_local_result_before_return_offer", true)
        }

        val localProofGeometry =
            if (houseSeatGeometry.length() > 0) {
                houseSeatGeometry
            } else {
                val seedLocalProofGeometry = JSONObject().apply {
                    if (!targetCell.isNullOrBlank()) {
                        put("geometry_kind", "CELL_THREE_HOUSE_UNIVERSE")
                        put("target_cell", targetCell)
                        put("asked_digit", targetDigit ?: JSONObject.NULL)
                        put("houses", JSONArray(listOfNotNull(houseScope)))
                        put("blocked_digits_by_house", JSONObject())
                        put("merged_blocked_digits", JSONArray())
                        put("surviving_digits", targetCellQuery.optJSONObject("target_before_state")?.optJSONArray("digits") ?: JSONArray())
                        put("candidate_status_map", JSONArray())
                        put("blocker_receipts", JSONArray())
                        put("scan_order", JSONArray(listOf("ROW", "COLUMN", "BOX")))
                        put("primary_spotlight", targetCell)
                    } else {
                        put("geometry_kind", "NONE")
                        put("primary_spotlight", JSONObject.NULL)
                    }
                }

                enrichCellThreeHouseGeometryFromBlockersV1(
                    seedGeometry = seedLocalProofGeometry,
                    narrativeGeometry = null,
                    targetCell = targetCell,
                    askedDigit = targetDigit,
                    houses = JSONArray(listOfNotNull(houseScope)),
                    targetBeforeState = targetCellQuery.optJSONObject("target_before_state"),
                    blockersPacket = blockers
                )
            }

        val visualLanguage = JSONObject().apply {
            put("may_use_spotlight_language", true)
            put("may_use_scene_language", true)
            put("may_use_actor_language", true)
            put("may_use_scan_language", true)
            put("may_use_survival_language", survivorSummary.length() > 0)
            put("may_use_duel_language", false)
            put("may_use_pattern_visibility_language", false)
            put("must_stay_grounded_in_packet_truth", true)
        }


        val speechBoundary = JSONObject().apply {
            put("must_answer_local_question_first", true)
            put("must_not_reopen_route", true)
            put("must_not_board_audit", true)
            put("must_not_commit", true)
            put("may_briefly_bridge_back", true)
            put("preferred_answer_length", "SHORT_TO_MEDIUM")
            put("end_with_handback", false)
        }

        val handbackContext = JSONObject().apply {
            put("paused_move_exists", true)
            put("handback_line", JSONObject.NULL)
            put("return_target_kind", "CURRENT_MOVE")
            put("return_target_label", JSONObject.NULL)
            put("may_offer_return_now", true)
            put("may_offer_one_followup", true)
            put("preferred_return_style", "GENTLE")
            put("preferred_followup_style", "BOUNDED")
        }

        val supportingFacts = JSONObject().apply {
            put(
                "focus_cells",
                if (houseSeatGeometry.length() > 0) {
                    houseSeatGeometry.optJSONArray("open_seats") ?: JSONArray(listOfNotNull(targetCell))
                } else {
                    JSONArray(listOfNotNull(targetCell))
                }
            )
            put("focus_houses", JSONArray(listOfNotNull(houseScope)))
            put(
                "candidate_snapshot",
                targetCellQuery.optJSONObject("target_before_state")
                    ?: proofChallenge.optJSONObject("target_before_state")
                    ?: JSONObject()
            )
            put(
                "house_candidate_map_excerpt",
                solverHouseMap.optJSONObject("result")
                    ?: solverHouseMap
            )
            put("technique_structure_excerpt", JSONObject())
        }

        val debugSupport = JSONObject().apply {
            put("fallback_reason", "normalized_detour_move_proof_missing")
            if (proofChallenge.length() > 0) put("proof_challenge_packet", proofChallenge)
            if (targetCellQuery.length() > 0) put("target_cell_query_packet", targetCellQuery)
            if (blockers.length() > 0) put("solver_cell_digit_blockers_packet", blockers)
            if (solverCellsCandidates.length() > 0) put("solver_cells_candidates_packet", solverCellsCandidates)
            if (solverHouseMap.length() > 0) put("solver_house_candidate_map_packet", solverHouseMap)
            if (scopedSupport.length() > 0) put("solver_scoped_support_packet", scopedSupport)
        }

        return ProofChallengePacketV1(
            challengeLane = effectiveChallengeLane,
            questionFrame = questionFrame,
            storyFocus = storyFocus,
            storyQuestion = storyQuestion,
            answerTruth = answerTruth,
            proofObject = JSONObject().apply {
                put("proof_object", questionFrame.opt("proof_object") ?: JSONObject.NULL)
                put("claim_kind", questionFrame.opt("proof_object") ?: JSONObject.NULL)
                put("challenge_lane", effectiveChallengeLane)
                put("elimination_kind", JSONObject.NULL)
            },
            proofMethod = proofMethod,

            narrativeArchetype = JSONObject().apply {
                val id = firstNonBlankStringV1(proofMethod.optString("narrative_archetype", null))
                    ?: archetypeForProof(
                        methodFamily = firstNonBlankStringV1(proofMethod.optString("method_family", null)),
                        proofObject = firstNonBlankStringV1(questionFrame.optString("proof_object", null)),
                        challengeLane = firstNonBlankStringV1(questionFrame.optString("challenge_lane", null)),
                        answerPolarity = firstNonBlankStringV1(answerTruth.optString("answer_polarity", null)),
                        nonproofReason = firstNonBlankStringV1(proofOutcome.optString("nonproof_reason", null)),
                        rivalCell = firstNonBlankStringV1(questionFrame.optString("contrast_cell", null)),
                        claimedTechniqueId = firstNonBlankStringV1(questionFrame.optString("claimed_technique_id", null))
                    )
                put("id", id)
                put("enum_name", packetArchetypeEnumV1(id).name)
                put(
                    "family",
                    when (id.trim().uppercase()) {
                        "LOCAL_CONTRADICTION_SPOTLIGHT" -> "LOCAL_CONTRADICTION"
                        "SURVIVOR_LADDER" -> "LOCAL_SURVIVOR"
                        "CONTRAST_DUEL" -> "LOCAL_CONTRAST"
                        "PATTERN_LEGITIMACY_CHECK" -> "PATTERN_LEGITIMACY"
                        "HONEST_INSUFFICIENCY_ANSWER" -> "HONEST_INSUFFICIENCY"
                        else -> "LOCAL_PROOF"
                    }
                )
            },
            doctrine = JSONObject().apply {
                val archetypeId = firstNonBlankStringV1(proofMethod.optString("narrative_archetype", null))
                    ?: archetypeForProof(
                        methodFamily = firstNonBlankStringV1(proofMethod.optString("method_family", null)),
                        proofObject = firstNonBlankStringV1(questionFrame.optString("proof_object", null)),
                        challengeLane = firstNonBlankStringV1(questionFrame.optString("challenge_lane", null)),
                        answerPolarity = firstNonBlankStringV1(answerTruth.optString("answer_polarity", null)),
                        nonproofReason = firstNonBlankStringV1(proofOutcome.optString("nonproof_reason", null)),
                        rivalCell = firstNonBlankStringV1(questionFrame.optString("contrast_cell", null)),
                        claimedTechniqueId = firstNonBlankStringV1(questionFrame.optString("claimed_technique_id", null))
                    )
                val id = firstNonBlankStringV1(proofMethod.optString("doctrine_id", null)) ?: doctrineForArchetype(archetypeId)
                put("id", id)
                put(
                    "answer_shape",
                    when (archetypeId.trim().uppercase()) {
                        "LOCAL_CONTRADICTION_SPOTLIGHT" -> "LOCAL_BLOCKER_EXPLANATION"
                        "SURVIVOR_LADDER" -> "SURVIVOR_PROOF"
                        "CONTRAST_DUEL" -> "RIVAL_COMPARISON"
                        "PATTERN_LEGITIMACY_CHECK" -> "PATTERN_VALIDATION"
                        else -> "HONEST_LIMITED_ANSWER"
                    }
                )
                put("must_answer_local_question_first", true)
                put("must_not_reopen_route", true)
                put("must_not_board_audit", true)
                put("must_not_commit", true)
                put("must_end_with_handback", false)
            },

            speechSkeleton = JSONArray(),
            actorModel = JSONObject().apply {
                put("id", "FALLBACK_LOCAL_SINGLE_SCOPE")
            },
            storyActors = storyActors,
            proofLadder = proofLadder,
            proofOutcome = proofOutcome,
            storyArc = storyArc,
            microStagePlan = microStagePlan,
            localProofGeometry = localProofGeometry,
            speechBoundary = speechBoundary,
            closureContract = closureContract,
            handbackContext = handbackContext,

            overlayPlan = JSONObject().apply {
                put("overlay_story_kind", JSONObject.NULL)
                put("overlay_mode", "REPLACE")
                put("primary_focus_cells", supportingFacts.optJSONArray("focus_cells") ?: JSONArray())
                put("primary_focus_houses", supportingFacts.optJSONArray("focus_houses") ?: JSONArray())
                put("secondary_focus_cells", JSONArray())
                put("deemphasize_cells", JSONArray())
                put("reason_for_focus", "fallback_move_proof_focus")
                put("highlight_roles", JSONObject())
            },

            visualLanguage = visualLanguage,
            supportingFacts = supportingFacts,
            debugSupport = debugSupport
        ).toJson()
    }

    fun projectDetourLocalGridInspectionPacket(replyRequest: ReplyRequestV1): JSONObject {
        val normalized = projectNormalizedDetourLocalInspection(replyRequest)
        if (normalized.length() > 0) {
            val anchor = normalized.optJSONObject("anchor") ?: JSONObject()
            val scope = normalized.optJSONObject("scope") ?: JSONObject()
            val question = normalized.optJSONObject("question") ?: JSONObject()
            val inspectionTruth = normalized.optJSONObject("inspection_truth") ?: JSONObject()
            val routeContext = normalized.optJSONObject("route_context") ?: JSONObject()
            val overlayContext = normalized.optJSONObject("overlay_context") ?: JSONObject()
            val supportMeta = normalized.optJSONObject("support") ?: JSONObject()

            val inspectionProfile = when ((normalized.optString("query_profile", "")).trim().uppercase()) {
                "HOUSE_CANDIDATE_MAP" -> DetourLocalGridInspectionProfileV1.HOUSE_CANDIDATE_MAP
                "DIGIT_LOCATIONS" -> DetourLocalGridInspectionProfileV1.DIGIT_LOCATIONS
                "LOCAL_EFFECTS" -> DetourLocalGridInspectionProfileV1.LOCAL_EFFECTS
                "NEARBY_CELL_STATUS" -> DetourLocalGridInspectionProfileV1.NEARBY_CELL_STATUS
                "TARGET_NEIGHBORHOOD" -> DetourLocalGridInspectionProfileV1.TARGET_NEIGHBORHOOD
                else -> DetourLocalGridInspectionProfileV1.CELL_CANDIDATES
            }

            val focusCells = stringListFromJsonArrayV1(scope.optJSONArray("cells"))
            val focusHouses = stringListFromJsonArrayV1(scope.optJSONArray("houses"))

            val overlayPolicy = DetourOverlayPolicyV1(
                overlayMode = when (overlayContext.optString("overlay_mode", "").trim().uppercase()) {
                    "REPLACE" -> DetourOverlayModeV1.REPLACE
                    "CLEAR" -> DetourOverlayModeV1.CLEAR
                    "PRESERVE" -> DetourOverlayModeV1.PRESERVE
                    else -> DetourOverlayModeV1.AUGMENT
                },
                primaryFocusCells = stringListFromJsonArrayV1(overlayContext.optJSONArray("focus_cells")),
                primaryFocusHouses = stringListFromJsonArrayV1(overlayContext.optJSONArray("focus_houses")),
                secondaryFocusCells = stringListFromJsonArrayV1(overlayContext.optJSONArray("secondary_focus_cells")),
                deemphasizeCells = stringListFromJsonArrayV1(overlayContext.optJSONArray("deemphasize_cells")),
                reasonForFocus = firstNonBlankStringV1(overlayContext.optString("reason_for_focus", null)),
                expectedSpokenAnchor = firstNonBlankStringV1(
                    question.optString("central_question", null),
                    focusCells.firstOrNull(),
                    focusHouses.firstOrNull()
                )
            )

            val handbackPolicy = DetourHandbackPolicyV1(
                handoverMode = parseDetourHandoverModeV1(
                    routeContext.optString("recommended_handover_mode", null)
                ),
                pausedRouteCheckpoint = anchor.optString("paused_route_checkpoint_id", "").ifBlank { null },
                returnTargetStage = anchor.optString("story_stage", "").ifBlank { null },
                returnTargetStepId = anchor.optString("step_id", "").ifBlank { null },
                stayDetachedUntilUserSaysContinue =
                    parseDetourHandoverModeV1(routeContext.optString("recommended_handover_mode", null)) ==
                            DetourHandoverModeV1.AWAIT_USER_CONTROL,
                spokenReturnLine = firstNonBlankStringV1(
                    routeContext.optString("spoken_return_line", null),
                    "That gives the local board readout; we can return to the paused move whenever you want."
                )
            )

            val support = JSONObject().apply {
                put("normalized_detour_local_inspection", normalized)
                if (supportMeta.length() > 0) put("normalizer_support", supportMeta)
            }


            val sharedCandidatesSummary =
                inspectionTruth.optJSONObject("shared_candidates_summary") ?: JSONObject()

            val directAnswerTruth = JSONObject().apply {
                put("answer_kind", "LOCAL_READOUT")
                put(
                    "short_answer",
                    when (inspectionProfile) {
                        DetourLocalGridInspectionProfileV1.CELL_CANDIDATES ->
                            firstNonBlankStringV1(
                                inspectionTruth.optJSONObject("candidate_state")?.optString("summary", null),
                                inspectionTruth.optJSONObject("candidate_state")?.optString("cell", null)?.let { cell ->
                                    val digits = inspectionTruth.optJSONObject("candidate_state")?.optJSONArray("digits")
                                    if (digits != null) "$cell currently has candidates ${digits}." else null
                                }
                            ) ?: JSONObject.NULL

                        DetourLocalGridInspectionProfileV1.DIGIT_LOCATIONS ->
                            firstNonBlankStringV1(question.optString("central_question", null)) ?: JSONObject.NULL

                        else ->
                            firstNonBlankStringV1(question.optString("central_question", null)) ?: JSONObject.NULL
                    }
                )
                put("compare_mode", inspectionProfile == DetourLocalGridInspectionProfileV1.NEARBY_CELL_STATUS && sharedCandidatesSummary.length() > 0)
                put("must_read_out_local_state_first", true)
                put("must_not_claim_missing_evidence_if_packet_has_readout", inspectionTruth.optJSONObject("candidate_state")?.length() ?: 0 > 0 || sharedCandidatesSummary.length() > 0)
                put("shared_candidates_summary", sharedCandidatesSummary)
            }

            val replyDiscipline = JSONObject().apply {
                put("answer_local_truth_first", true)
                put("route_summary_secondary", true)
                put("forbid_main_route_reframing_before_answer", true)
                put("forbid_missing_evidence_claim_if_packet_has_local_truth", true)
                put("prefer_packet_readout_over_generic_state_uncertainty", true)
            }



            val localInspectionNarrativeSurface =
                detourNarrativeSurfaceV1(
                    replyRequest = replyRequest,
                    doctrineFamily = "local_grid_inspection_doctrine",
                    defaultAnswerShape = "scope -> state readout -> why it matters -> bounded handback",
                    defaultOrderedExplanationLadder = listOf(
                        "scope",
                        "state_readout",
                        "why_it_matters",
                        "bounded_handback"
                    ),
                    defaultBoundaryLine = "Keep this as a bounded local readout. Do not expand into a full proof ladder.",
                    defaultHandbackLine = "That gives the local board readout; we can return to the paused move whenever you want."
                )

            return DetourLocalGridInspectionPacketV1(
                inspectionProfile = inspectionProfile,
                anchorStepId = firstNonBlankStringV1(anchor.optString("step_id", null)),
                scopeKind = firstNonBlankStringV1(scope.optString("kind", null)),
                scopeRef = firstNonBlankStringV1(scope.optString("ref", null)),
                focusCells = focusCells,
                focusHouses = focusHouses,
                candidateState = inspectionTruth.optJSONObject("candidate_state") ?: JSONObject(),
                digitLocations = inspectionTruth.optJSONArray("digit_locations") ?: JSONArray(),
                localDelta = inspectionTruth.optJSONObject("local_delta") ?: JSONObject(),

                nearbyEffectsSummary = inspectionTruth.optJSONObject("nearby_effects_summary") ?: JSONObject(),
                directAnswerTruth = directAnswerTruth,
                replyDiscipline = replyDiscipline,
                doctrineSurface = localInspectionNarrativeSurface.optJSONObject("doctrine_surface") ?: JSONObject(),

                answerShape = firstNonBlankStringV1(localInspectionNarrativeSurface.optString("answer_shape", null)),
                orderedExplanationLadder = localInspectionNarrativeSurface.optJSONArray("ordered_explanation_ladder") ?: JSONArray(),
                boundaryLine = firstNonBlankStringV1(localInspectionNarrativeSurface.optString("boundary_line", null)),
                handbackLine = firstNonBlankStringV1(localInspectionNarrativeSurface.optString("handback_line", null)),
                support = support,
                overlayPolicy = overlayPolicy,
                handbackPolicy = handbackPolicy,
                answerBoundary = listOf(
                    DetourAnswerBoundaryV1.DO_NOT_BECOME_PROOF_LADDER,
                    DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                    DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
                )
            ).toJson()
        }

        val detourQuestionClass = activeDetourQuestionClassFromReplyRequestV1(replyRequest)
            ?.trim()
            ?.uppercase()

        if (
            detourQuestionClass != "NEIGHBOR_CELL_QUERY" &&
            detourQuestionClass != "CANDIDATE_STATE_QUERY" &&
            detourQuestionClass != "TARGET_CELL_QUERY"
        ) return JSONObject()

        val candidateState = projectCandidateStatePacket(replyRequest)
        val neighborCellQuery = projectNeighborCellQueryPacket(replyRequest)
        val targetCellQuery = projectTargetCellQueryPacket(replyRequest)
        val solverCellCandidates = projectSolverCellCandidatesPacket(replyRequest)
        val solverCellsCandidates = projectSolverCellsCandidatesPacket(replyRequest)
        val solverHouseMap = projectSolverHouseCandidateMapPacket(replyRequest)

        if (
            candidateState.length() == 0 &&
            neighborCellQuery.length() == 0 &&
            targetCellQuery.length() == 0 &&
            solverCellCandidates.length() == 0 &&
            solverCellsCandidates.length() == 0 &&
            solverHouseMap.length() == 0
        ) return JSONObject()

        val focusCells =
            buildList {
                firstNonBlankStringV1(
                    candidateState.optString("cell", null),
                    neighborCellQuery.optString("neighbor_cell", null),
                    neighborCellQuery.optString("cell", null),
                    targetCellQuery.optString("target_cell", null),
                    solverCellCandidates.optString("cell", null)
                )?.let { add(it) }
                addAll(stringListFromJsonArrayV1(candidateState.optJSONArray("cells")))
            }.distinct()

        val focusHouses =
            buildList {
                firstNonBlankStringV1(
                    candidateState.optString("house_scope", null),
                    neighborCellQuery.optString("house_scope", null),
                    targetCellQuery.optString("house_scope", null),
                    solverHouseMap.optString("house_scope", null)
                )?.let { add(it) }
            }.distinct()

        val inspectionProfile =
            when {
                solverHouseMap.length() > 0 -> DetourLocalGridInspectionProfileV1.HOUSE_CANDIDATE_MAP
                detourQuestionClass == "NEIGHBOR_CELL_QUERY" -> DetourLocalGridInspectionProfileV1.NEARBY_CELL_STATUS
                detourQuestionClass == "TARGET_CELL_QUERY" -> DetourLocalGridInspectionProfileV1.TARGET_NEIGHBORHOOD
                else -> DetourLocalGridInspectionProfileV1.CELL_CANDIDATES
            }

        val support = JSONObject().apply {
            if (candidateState.length() > 0) put("candidate_state_packet", candidateState)
            if (neighborCellQuery.length() > 0) put("neighbor_cell_query_packet", neighborCellQuery)
            if (targetCellQuery.length() > 0) put("target_cell_query_packet", targetCellQuery)
            if (solverCellCandidates.length() > 0) put("solver_cell_candidates_packet", solverCellCandidates)
            if (solverCellsCandidates.length() > 0) put("solver_cells_candidates_packet", solverCellsCandidates)
            if (solverHouseMap.length() > 0) put("solver_house_candidate_map_packet", solverHouseMap)
            put("fallback_reason", "normalized_detour_local_inspection_missing")
        }

        return DetourLocalGridInspectionPacketV1(
            inspectionProfile = inspectionProfile,
            anchorStepId = firstNonBlankStringV1(
                candidateState.optString("step_id", null),
                neighborCellQuery.optString("step_id", null),
                targetCellQuery.optString("step_id", null)
            ),
            scopeKind = firstNonBlankStringV1(
                candidateState.optString("ask_kind", null),
                neighborCellQuery.optString("scope_kind", null),
                solverHouseMap.optString("scope_kind", null)
            ),
            scopeRef = firstNonBlankStringV1(
                candidateState.optString("scope_ref", null),
                neighborCellQuery.optString("scope_ref", null),
                solverHouseMap.optString("scope_ref", null)
            ),
            focusCells = focusCells,
            focusHouses = focusHouses,
            candidateState =
                if (candidateState.length() > 0) candidateState
                else if (solverCellCandidates.length() > 0) solverCellCandidates
                else JSONObject(),
            digitLocations =
                solverHouseMap.optJSONArray("digit_locations")
                    ?: candidateState.optJSONArray("digit_locations")
                    ?: JSONArray(),
            localDelta =
                neighborCellQuery.optJSONObject("local_delta")
                    ?: candidateState.optJSONObject("local_delta")
                    ?: JSONObject(),

            nearbyEffectsSummary =
                neighborCellQuery.optJSONObject("nearby_effects_summary")
                    ?: targetCellQuery.optJSONObject("nearby_effects_summary")
                    ?: JSONObject(),
            directAnswerTruth = JSONObject().apply {
                put("answer_kind", "LOCAL_READOUT")
                put(
                    "short_answer",
                    firstNonBlankStringV1(
                        candidateState.optString("summary", null),
                        neighborCellQuery.optString("summary", null),
                        targetCellQuery.optString("summary", null)
                    ) ?: JSONObject.NULL
                )
                put("must_read_out_local_state_first", true)
                put("must_not_claim_missing_evidence_if_packet_has_readout", candidateState.length() > 0 || solverCellCandidates.length() > 0 || solverHouseMap.length() > 0)
            },
            replyDiscipline = JSONObject().apply {
                put("answer_local_truth_first", true)
                put("route_summary_secondary", true)
                put("forbid_main_route_reframing_before_answer", true)
                put("forbid_missing_evidence_claim_if_packet_has_local_truth", true)
            },
            support = support,
            overlayPolicy = projectWave1DetourOverlayPolicyV1(


                focusCells = focusCells,
                focusHouses = focusHouses,
                reasonForFocus = "wave1_local_grid_inspection_fallback",
                spokenAnchor = focusCells.firstOrNull() ?: focusHouses.firstOrNull()
            ),
            handbackPolicy = projectWave1HandbackPolicyV1(
                replyRequest,
                spokenReturnLine = "That gives the local board readout; we can return to the paused move whenever you want."
            )
        ).toJson()
    }

    fun projectDetourUserProposalVerdictPacket(replyRequest: ReplyRequestV1): JSONObject {
        val normalized = projectNormalizedDetourProposalVerdict(replyRequest)
        if (normalized.length() > 0) {
            val anchor = normalized.optJSONObject("anchor") ?: JSONObject()
            val scope = normalized.optJSONObject("scope") ?: JSONObject()
            val question = normalized.optJSONObject("question") ?: JSONObject()
            val proposalTruth = normalized.optJSONObject("proposal_truth") ?: JSONObject()
            val routeContext = normalized.optJSONObject("route_context") ?: JSONObject()
            val overlayContext = normalized.optJSONObject("overlay_context") ?: JSONObject()
            val supportMeta = normalized.optJSONObject("support") ?: JSONObject()

            val proposalKind = when ((proposalTruth.optString("proposal_kind", "")).trim().uppercase()) {
                "PROPOSED_DIGIT" -> DetourUserProposalKindV1.PROPOSED_DIGIT
                "PROPOSED_ELIMINATION" -> DetourUserProposalKindV1.PROPOSED_ELIMINATION
                "PROPOSED_PATTERN" -> DetourUserProposalKindV1.PROPOSED_PATTERN
                "PROPOSED_HOUSE_CLAIM" -> DetourUserProposalKindV1.PROPOSED_HOUSE_CLAIM
                "PROPOSED_LOCAL_CHAIN" -> DetourUserProposalKindV1.PROPOSED_LOCAL_CHAIN
                else -> DetourUserProposalKindV1.GENERAL_REASONING_CHECK
            }

            val overlayPolicy = DetourOverlayPolicyV1(
                overlayMode = when (overlayContext.optString("overlay_mode", "").trim().uppercase()) {
                    "REPLACE" -> DetourOverlayModeV1.REPLACE
                    "CLEAR" -> DetourOverlayModeV1.CLEAR
                    "PRESERVE" -> DetourOverlayModeV1.PRESERVE
                    else -> DetourOverlayModeV1.AUGMENT
                },
                primaryFocusCells = stringListFromJsonArrayV1(overlayContext.optJSONArray("focus_cells")),
                primaryFocusHouses = stringListFromJsonArrayV1(overlayContext.optJSONArray("focus_houses")),
                secondaryFocusCells = stringListFromJsonArrayV1(overlayContext.optJSONArray("secondary_focus_cells")),
                deemphasizeCells = stringListFromJsonArrayV1(overlayContext.optJSONArray("deemphasize_cells")),
                reasonForFocus = firstNonBlankStringV1(overlayContext.optString("reason_for_focus", null)),
                expectedSpokenAnchor = firstNonBlankStringV1(
                    question.optString("proposal_summary", null),
                    question.optString("proposal_text", null),
                    scope.optString("ref", null)
                )
            )

            val handbackPolicy = DetourHandbackPolicyV1(
                handoverMode = parseDetourHandoverModeV1(
                    routeContext.optString("recommended_handover_mode", null)
                ),
                pausedRouteCheckpoint = anchor.optString("paused_route_checkpoint_id", "").ifBlank { null },
                returnTargetStage = anchor.optString("story_stage", "").ifBlank { null },
                returnTargetStepId = anchor.optString("step_id", "").ifBlank { null },
                stayDetachedUntilUserSaysContinue =
                    parseDetourHandoverModeV1(routeContext.optString("recommended_handover_mode", null)) ==
                            DetourHandoverModeV1.AWAIT_USER_CONTROL,
                spokenReturnLine = firstNonBlankStringV1(
                    routeContext.optString("spoken_return_line", null),
                    "That checks your idea against the current board truth; the paused route is still available."
                )
            )

            val whatIsCorrect = JSONArray().apply {
                val arr = proposalTruth.optJSONArray("what_is_correct") ?: JSONArray()
                for (i in 0 until arr.length()) put(arr.get(i))
            }

            val whatIsIncorrect = JSONArray().apply {
                val arr = proposalTruth.optJSONArray("what_is_incorrect") ?: JSONArray()
                for (i in 0 until arr.length()) put(arr.get(i))
            }

            val support = JSONObject().apply {
                put("normalized_detour_proposal_verdict", normalized)
                if (supportMeta.length() > 0) put("normalizer_support", supportMeta)
            }

            val proposalVerdictNarrativeSurface =
                detourNarrativeSurfaceV1(
                    replyRequest = replyRequest,
                    doctrineFamily = "proposal_verdict_doctrine",
                    defaultAnswerShape = "verdict -> what works -> what fails or is missing -> route relation -> bounded handback",
                    defaultOrderedExplanationLadder = listOf(
                        "verdict",
                        "what_works",
                        "what_fails_or_is_missing",
                        "route_relation",
                        "bounded_handback"
                    ),
                    defaultBoundaryLine = "Give the verdict and bounded reasoning only. Do not switch routes or turn this into a board audit.",
                    defaultHandbackLine = "That checks your idea against the current board truth; the paused route is still available."
                )

            return DetourUserProposalVerdictPacketV1(
                proposalKind = proposalKind,
                proposalText = firstNonBlankStringV1(
                    question.optString("proposal_text", null),
                    question.optString("proposal_summary", null)
                ),
                proposalScope = firstNonBlankStringV1(
                    scope.optString("ref", null),
                    scope.optString("kind", null)
                ),
                verdict = parseReasoningVerdictV1(proposalTruth.optString("verdict", null)),
                verdictReason = firstNonBlankStringV1(proposalTruth.optString("verdict_reason", null)),
                whatIsCorrect = whatIsCorrect,
                whatIsIncorrect = whatIsIncorrect,
                missingCondition = firstNonBlankStringV1(proposalTruth.optString("missing_condition", null)),
                routeAlignment = firstNonBlankStringV1(proposalTruth.optString("route_alignment", null)),
                anchorStepId = firstNonBlankStringV1(anchor.optString("step_id", null)),
                anchorStoryStage = firstNonBlankStringV1(anchor.optString("story_stage", null)),
                solverSupportRows = proposalTruth.optJSONArray("support_rows") ?: JSONArray(),
                doctrineSurface = proposalVerdictNarrativeSurface.optJSONObject("doctrine_surface") ?: JSONObject(),
                answerShape = firstNonBlankStringV1(proposalVerdictNarrativeSurface.optString("answer_shape", null)),
                orderedExplanationLadder = proposalVerdictNarrativeSurface.optJSONArray("ordered_explanation_ladder") ?: JSONArray(),
                boundaryLine = firstNonBlankStringV1(proposalVerdictNarrativeSurface.optString("boundary_line", null)),
                handbackLine = firstNonBlankStringV1(proposalVerdictNarrativeSurface.optString("handback_line", null)),
                support = support,
                overlayPolicy = overlayPolicy,
                handbackPolicy = handbackPolicy,
                answerBoundary = listOf(
                    DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
                    DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
                    DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
                )
            ).toJson()
        }

        val detourQuestionClass = activeDetourQuestionClassFromReplyRequestV1(replyRequest)
            ?.trim()
            ?.uppercase()

        if (detourQuestionClass != "USER_REASONING_CHECK") return JSONObject()

        val userReasoningCheck = projectUserReasoningCheckPacket(replyRequest)
        val solverReasoningCheck = projectSolverReasoningCheckPacket(replyRequest)
        val scopedSupport = projectSolverScopedSupportPacket(replyRequest)

        if (
            userReasoningCheck.length() == 0 &&
            solverReasoningCheck.length() == 0 &&
            scopedSupport.length() == 0
        ) return JSONObject()

        val verdict = parseReasoningVerdictV1(
            firstNonBlankStringV1(
                solverReasoningCheck.optString("verdict", null),
                userReasoningCheck.optString("verdict", null)
            )
        )

        val support = JSONObject().apply {
            if (userReasoningCheck.length() > 0) put("user_reasoning_check_packet", userReasoningCheck)
            if (solverReasoningCheck.length() > 0) put("solver_reasoning_check_packet", solverReasoningCheck)
            if (scopedSupport.length() > 0) put("solver_scoped_support_packet", scopedSupport)
            put("fallback_reason", "normalized_detour_proposal_verdict_missing")
        }

        val whatIsCorrect =
            solverReasoningCheck.optJSONArray("what_is_correct")
                ?: userReasoningCheck.optJSONArray("what_is_correct")
                ?: JSONArray()

        val whatIsIncorrect =
            solverReasoningCheck.optJSONArray("what_is_incorrect")
                ?: userReasoningCheck.optJSONArray("what_is_incorrect")
                ?: JSONArray()

        return DetourUserProposalVerdictPacketV1(
            proposalKind = when (
                firstNonBlankStringV1(
                    userReasoningCheck.optString("proposal_kind", null),
                    solverReasoningCheck.optString("proposal_kind", null)
                )?.uppercase()
            ) {
                "PROPOSED_DIGIT" -> DetourUserProposalKindV1.PROPOSED_DIGIT
                "PROPOSED_ELIMINATION" -> DetourUserProposalKindV1.PROPOSED_ELIMINATION
                "PROPOSED_PATTERN" -> DetourUserProposalKindV1.PROPOSED_PATTERN
                "PROPOSED_HOUSE_CLAIM" -> DetourUserProposalKindV1.PROPOSED_HOUSE_CLAIM
                "PROPOSED_LOCAL_CHAIN" -> DetourUserProposalKindV1.PROPOSED_LOCAL_CHAIN
                else -> DetourUserProposalKindV1.GENERAL_REASONING_CHECK
            },
            proposalText = firstNonBlankStringV1(
                userReasoningCheck.optString("proposal_text", null),
                userReasoningCheck.optString("user_reasoning", null),
                solverReasoningCheck.optString("proposal_text", null)
            ),
            proposalScope = firstNonBlankStringV1(
                userReasoningCheck.optString("proposal_scope", null),
                solverReasoningCheck.optString("proposal_scope", null)
            ),
            verdict = verdict,
            verdictReason = firstNonBlankStringV1(
                solverReasoningCheck.optString("verdict_reason", null),
                userReasoningCheck.optString("verdict_reason", null)
            ),
            whatIsCorrect = whatIsCorrect,
            whatIsIncorrect = whatIsIncorrect,
            missingCondition = firstNonBlankStringV1(
                solverReasoningCheck.optString("missing_condition", null),
                userReasoningCheck.optString("missing_condition", null)
            ),
            routeAlignment = firstNonBlankStringV1(
                solverReasoningCheck.optString("route_alignment", null),
                userReasoningCheck.optString("route_alignment", null)
            ),
            anchorStepId = firstNonBlankStringV1(
                userReasoningCheck.optString("step_id", null),
                solverReasoningCheck.optString("step_id", null)
            ),
            anchorStoryStage = firstNonBlankStringV1(
                userReasoningCheck.optString("story_stage", null),
                solverReasoningCheck.optString("story_stage", null)
            ),
            solverSupportRows =
                solverReasoningCheck.optJSONArray("support_rows")
                    ?: scopedSupport.optJSONArray("bounded_rows")
                    ?: JSONArray(),
            support = support,
            overlayPolicy = projectWave1DetourOverlayPolicyV1(
                focusCells = buildList {
                    firstNonBlankStringV1(
                        userReasoningCheck.optString("target_cell", null),
                        solverReasoningCheck.optString("target_cell", null)
                    )?.let { add(it) }
                },
                reasonForFocus = "wave1_user_proposal_verdict_fallback",
                spokenAnchor = firstNonBlankStringV1(
                    userReasoningCheck.optString("target_cell", null),
                    solverReasoningCheck.optString("target_cell", null)
                )
            ),
            handbackPolicy = projectWave1HandbackPolicyV1(
                replyRequest,
                spokenReturnLine = "That checks your idea against the current board truth; the paused route is still available."
            )
        ).toJson()
    }

    private fun appendPhase2DetourPacketsV1(
        out: JSONArray,
        replyRequest: ReplyRequestV1
    ) {
        val stepClarification = projectStepClarificationPacket(replyRequest)
        if (stepClarification.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "STEP_CLARIFICATION_PACKET")
                put("payload", stepClarification)
            })
        }

        val proofChallenge = projectProofChallengePacket(replyRequest)
        if (proofChallenge.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "PROOF_CHALLENGE_PACKET")
                put("payload", proofChallenge)
            })
        }

        val userReasoningCheck = projectUserReasoningCheckPacket(replyRequest)
        if (userReasoningCheck.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "USER_REASONING_CHECK_PACKET")
                put("payload", userReasoningCheck)
            })
        }

        val alternativeTechnique = projectAlternativeTechniquePacket(replyRequest)
        if (alternativeTechnique.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "ALTERNATIVE_TECHNIQUE_PACKET")
                put("payload", alternativeTechnique)
            })
        }

        val solverCellCandidates = projectSolverCellCandidatesPacket(replyRequest)
        if (solverCellCandidates.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_CELL_CANDIDATES_PACKET")
                put("payload", solverCellCandidates)
            })
        }

        val solverCellsCandidates = projectSolverCellsCandidatesPacket(replyRequest)
        if (solverCellsCandidates.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_CELLS_CANDIDATES_PACKET")
                put("payload", solverCellsCandidates)
            })
        }

        val solverHouseCandidateMap = projectSolverHouseCandidateMapPacket(replyRequest)
        if (solverHouseCandidateMap.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_HOUSE_CANDIDATE_MAP_PACKET")
                put("payload", solverHouseCandidateMap)
            })
        }

        val solverCellDigitBlockers = projectSolverCellDigitBlockersPacket(replyRequest)
        if (solverCellDigitBlockers.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_CELL_DIGIT_BLOCKERS_PACKET")
                put("payload", solverCellDigitBlockers)
            })
        }

        val solverReasoningCheck = projectSolverReasoningCheckPacket(replyRequest)
        if (solverReasoningCheck.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_REASONING_CHECK_PACKET")
                put("payload", solverReasoningCheck)
            })
        }

        val solverAlternativeTechnique = projectSolverAlternativeTechniquePacket(replyRequest)
        if (solverAlternativeTechnique.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_ALTERNATIVE_TECHNIQUE_PACKET")
                put("payload", solverAlternativeTechnique)
            })
        }

        val solverTechniqueScopeCheck = projectSolverTechniqueScopeCheckPacket(replyRequest)
        if (solverTechniqueScopeCheck.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET")
                put("payload", solverTechniqueScopeCheck)
            })
        }

        val solverLocalMoveSearch = projectSolverLocalMoveSearchPacket(replyRequest)
        if (solverLocalMoveSearch.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_LOCAL_MOVE_SEARCH_PACKET")
                put("payload", solverLocalMoveSearch)
            })
        }

        val solverRouteComparison = projectSolverRouteComparisonPacket(replyRequest)
        if (solverRouteComparison.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_ROUTE_COMPARISON_PACKET")
                put("payload", solverRouteComparison)
            })
        }

        val solverScopedSupport = projectSolverScopedSupportPacket(replyRequest)
        if (solverScopedSupport.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_SCOPED_SUPPORT_PACKET")
                put("payload", solverScopedSupport)
            })
        }

        val targetCellQuery = projectTargetCellQueryPacket(replyRequest)
        if (targetCellQuery.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "TARGET_CELL_QUERY_PACKET")
                put("payload", targetCellQuery)
            })
        }

        val candidateState = projectCandidateStatePacket(replyRequest)
        if (candidateState.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "CANDIDATE_STATE_PACKET")
                put("payload", candidateState)
            })
        }

        val neighborCellQuery = projectNeighborCellQueryPacket(replyRequest)
        if (neighborCellQuery.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "NEIGHBOR_CELL_QUERY_PACKET")
                put("payload", neighborCellQuery)
            })
        }

        val returnToRoute = projectReturnToRoutePacket(replyRequest)
        if (returnToRoute.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "RETURN_TO_ROUTE_PACKET")
                put("payload", returnToRoute)
            })
        }
    }

    // ---------------------------------------------------------------------
    // Phase 7 — detour payload discipline
    // ---------------------------------------------------------------------

    private fun hasAnyDetourPacketV1(replyRequest: ReplyRequestV1): Boolean =
        projectStepClarificationPacket(replyRequest).length() > 0 ||
                projectProofChallengePacket(replyRequest).length() > 0 ||
                projectUserReasoningCheckPacket(replyRequest).length() > 0 ||
                projectAlternativeTechniquePacket(replyRequest).length() > 0 ||
                projectSolverCellCandidatesPacket(replyRequest).length() > 0 ||
                projectSolverCellsCandidatesPacket(replyRequest).length() > 0 ||
                projectSolverHouseCandidateMapPacket(replyRequest).length() > 0 ||
                projectSolverCellDigitBlockersPacket(replyRequest).length() > 0 ||
                projectSolverReasoningCheckPacket(replyRequest).length() > 0 ||
                projectSolverAlternativeTechniquePacket(replyRequest).length() > 0 ||
                projectSolverTechniqueScopeCheckPacket(replyRequest).length() > 0 ||
                projectSolverLocalMoveSearchPacket(replyRequest).length() > 0 ||
                projectSolverRouteComparisonPacket(replyRequest).length() > 0 ||
                projectSolverScopedSupportPacket(replyRequest).length() > 0 ||
                projectTargetCellQueryPacket(replyRequest).length() > 0 ||
                projectCandidateStatePacket(replyRequest).length() > 0 ||
                projectNeighborCellQueryPacket(replyRequest).length() > 0 ||
                projectReturnToRoutePacket(replyRequest).length() > 0


    private fun shouldUseBridgeFallbackOnlyV1(replyRequest: ReplyRequestV1): Boolean =
        isUserAgendaBridgeTurnV1(replyRequest) && !hasAnyDetourPacketV1(replyRequest)

    private fun isUserAgendaBridgeTurnV1(replyRequest: ReplyRequestV1): Boolean =
        replyRequest.turn.pendingAfter?.contains("UserAgendaBridge", ignoreCase = true) == true

    private fun buildUserAgendaBridgeFactsV1(replyRequest: ReplyRequestV1): JSONArray {


        if (!shouldUseBridgeFallbackOnlyV1(replyRequest)) {
            return buildDetourOnlyFactsV1(replyRequest)
        }

        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "DECISION_SUMMARY_MINI"); put("payload", projectDecisionSummaryMini(replyRequest)) })

            val cta = projectCtaContext(replyRequest)
            if (cta.length() > 0) {
                put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", cta) })
            }

            if (replyRequest.turn.phase == "CONFIRMING" || replyRequest.turn.phase == "SEALING") {
                val solvability = projectSolvabilityContext(replyRequest)
                if (solvability.length() > 0) {
                    put(JSONObject().apply { put("type", "SOLVABILITY_CONTEXT"); put("payload", solvability) })
                }

                val confirming = projectConfirmingContext(replyRequest)
                if (confirming.length() > 0) {
                    put(JSONObject().apply { put("type", "CONFIRMING_CONTEXT"); put("payload", confirming) })
                }
            }

            // Phase 7: detour packet builders are now the canonical supply for user-owned detour turns.
            appendCanonicalUserDetourPacketsV1(this, replyRequest)

            val glossary = projectGlossaryMini(replyRequest)
            if (glossary.length() > 0) {
                put(JSONObject().apply { put("type", "GLOSSARY_MINI"); put("payload", glossary) })
            }

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }
        }
    }


    private fun projectDetourRouteReturnPacket(replyRequest: ReplyRequestV1): JSONObject {
        if (!replyRequest.turn.turnRouteReturnAllowed) return JSONObject()
        if (replyRequest.turn.turnAuthorityOwner == "USER_ROUTE_JUMP_OWNER") return JSONObject()
        if (replyRequest.turn.turnBoundaryStatus == "RELEASED_TO_NEXT_STEP") return JSONObject()

        val turnJson = JSONObject(replyRequest.toJsonString()).optJSONObject("turn") ?: return JSONObject()
        val checkpoint = turnJson.optJSONObject("route_checkpoint_after") ?: return JSONObject()

        return JSONObject().apply {
            put("route_id", checkpoint.optString("route_id"))
            put("phase", checkpoint.optString("phase"))
            put("app_agenda_kind", checkpoint.optString("app_agenda_kind"))
            put("step_id", checkpoint.optString("step_id"))
            put("story_stage", checkpoint.optString("story_stage"))
            put("resume_prompt_hint", checkpoint.optString("resume_prompt_hint"))
        }
    }

    private fun appendCanonicalUserDetourPacketsV1(
        out: JSONArray,
        replyRequest: ReplyRequestV1
    ) {
        appendPhase2DetourPacketsV1(out, replyRequest)

        val support = projectSolverScopedSupportPacket(replyRequest)
        if (support.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_SCOPED_SUPPORT_PACKET")
                put("payload", support)
            })
        }
    }



    private fun preferredLocalDetourPacketsV1(
        replyRequest: ReplyRequestV1
    ): JSONArray {
        val out = JSONArray()

        val neighbor = projectNeighborCellQueryPacket(replyRequest)
        if (neighbor.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "NEIGHBOR_CELL_QUERY_PACKET")
                put("payload", neighbor)
            })
        }

        val target = projectTargetCellQueryPacket(replyRequest)
        if (target.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "TARGET_CELL_QUERY_PACKET")
                put("payload", target)
            })
        }

        val candidate = projectCandidateStatePacket(replyRequest)
        if (candidate.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "CANDIDATE_STATE_PACKET")
                put("payload", candidate)
            })
        }

        val blockers = projectSolverCellDigitBlockersPacket(replyRequest)
        if (blockers.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_CELL_DIGIT_BLOCKERS_PACKET")
                put("payload", blockers)
            })
        }

        val cellCandidates = projectSolverCellCandidatesPacket(replyRequest)
        if (cellCandidates.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_CELL_CANDIDATES_PACKET")
                put("payload", cellCandidates)
            })
        }

        val scopedSupport = projectSolverScopedSupportPacket(replyRequest)
        if (scopedSupport.length() > 0) {
            out.put(JSONObject().apply {
                put("type", "SOLVER_SCOPED_SUPPORT_PACKET")
                put("payload", scopedSupport)
            })
        }

        return out
    }




    private fun buildDetourOnlyFactsV1(replyRequest: ReplyRequestV1): JSONArray {
        return JSONArray().apply {
            put(JSONObject().apply {
                put("type", "STYLE_MINI")
                put("payload", projectStyleMini(replyRequest))
            })

            appendCanonicalUserDetourPacketsV1(this, replyRequest)

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0) {
                put(JSONObject().apply {
                    put("type", "CONTINUITY_SHORT")
                    put("payload", continuity)
                })
            }
        }
    }

    // ---------------------------------------------------------------------
    // Live projected fact arrays
    // ---------------------------------------------------------------------


    private fun activeDetourQuestionClassFromReplyRequestV1(
        replyRequest: ReplyRequestV1
    ): String? {
        return runCatching {
            val root = JSONObject(replyRequest.toJsonString())
            root.optJSONObject("turn")
                ?.optJSONObject("detour")
                ?.optString("detour_question_class", null)
        }.getOrNull()
    }

    private fun canonicalPrimaryDetourPacketTypesFromReplyRequestV1(
        detourQuestionClass: String?,
        facts: List<FactBundleV1>
    ): Set<FactBundleV1.Type> {
        val factTypes = facts.map { it.type }.toSet()

        if (FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1)
        }

        if (FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1)
        }

        if (FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1)
        }

        if (FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1)
        }

        if (FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1)
        }

        if (FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1)
        }

        if (FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1)
        }

        if (FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1)
        }

        if (FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes) {
            return setOf(FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1)
        }

        return when (detourQuestionClass?.trim()?.uppercase()) {
            "STEP_CLARIFICATION" -> setOf(
                FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1
            )

            "PROOF_CHALLENGE" -> setOf(
                FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1
            )

            "TARGET_CELL_QUERY" -> setOf(
                FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1
            )

            "NEIGHBOR_CELL_QUERY" -> setOf(
                FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1
            )

            "CANDIDATE_STATE_QUERY" -> setOf(
                FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1
            )

            "USER_REASONING_CHECK" -> setOf(
                FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1
            )

            "ALTERNATIVE_TECHNIQUE",
            "ALTERNATIVE_TECHNIQUE_QUERY" -> when {
                FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes ->
                    setOf(FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1)

                FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes ->
                    setOf(FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1)

                else ->
                    emptySet()
            }

            "ROUTE_COMPARISON",
            "ROUTE_COMPARISON_QUERY" -> setOf(
                FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1
            )

            "RETURN_TO_ROUTE" -> setOf(
                FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1
            )

            else -> when {
                FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes ->
                    setOf(FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1)

                FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1 in factTypes ->
                    setOf(FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1)

                else ->
                    emptySet()
            }
        }
    }

    private fun preferredDetourPacketTypesFromReplyRequestV1(
        detourQuestionClass: String?,
        pendingIsUserAgendaBridge: Boolean,
        allowsReturnToRoute: Boolean,
        facts: List<FactBundleV1>
    ): Set<FactBundleV1.Type> {
        val canonicalPrimary =
            canonicalPrimaryDetourPacketTypesFromReplyRequestV1(
                detourQuestionClass = detourQuestionClass,
                facts = facts
            )

        val support = when {
            FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1,
                FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1
            )

            FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1 in canonicalPrimary ||
                    FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1 in canonicalPrimary -> setOf(
                FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1
            )

            else -> emptySet()
        }

        val base = canonicalPrimary + support

        return if (
            pendingIsUserAgendaBridge &&
            allowsReturnToRoute &&
            detourQuestionClass?.trim()?.uppercase() != "RETURN_TO_ROUTE"
        ) {
            base + setOf(FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1)
        } else {
            base
        }
    }


    private fun hasProjectedDetourPacketV1(facts: List<FactBundleV1>): Boolean {
        return facts.any {
            when (it.type) {
                FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1,
                FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1,
                FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1,
                FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1,
                FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1,
                FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1,
                FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1,
                FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1,
                FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
                FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
                FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1,
                FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1,
                FactBundleV1.Type.SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET_V1,
                FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1,
                FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1,
                FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1 -> true

                else -> false
            }
        }
    }


    fun projectFactsForUserDetourDemand(replyRequest: ReplyRequestV1): JSONArray {
        val out = JSONArray()

        val turn = replyRequest.turn
        val bridgeFacts = buildUserAgendaBridgeFactsV1(replyRequest)
        val allowsReturnToRoute =
            bridgeFacts.toString().contains("RETURN_TO_ROUTE", ignoreCase = true)

        val projectedFacts = buildDetourOnlyFactsV1(replyRequest)

        val projectedTypeNames =
            buildList<String> {
                for (i in 0 until projectedFacts.length()) {
                    val item = projectedFacts.optJSONObject(i) ?: continue
                    val type = item.optString("type", "").trim()
                    if (type.isNotEmpty()) add(type)
                }
            }

        val projectedTypeSet = projectedTypeNames.toSet()

        val authoritativeDetourQuestionClass =
            when {
                "DETOUR_MOVE_PROOF_PACKET" in projectedTypeSet ->
                    "PROOF_CHALLENGE"

                "DETOUR_LOCAL_GRID_INSPECTION_PACKET" in projectedTypeSet ->
                    "NEIGHBOR_CELL_QUERY"

                "DETOUR_USER_PROPOSAL_VERDICT_PACKET" in projectedTypeSet ->
                    "USER_REASONING_CHECK"

                "DETOUR_ALTERNATIVE_TECHNIQUE_PACKET" in projectedTypeSet ->
                    "ALTERNATIVE_TECHNIQUE_QUERY"

                "DETOUR_LOCAL_MOVE_SEARCH_PACKET" in projectedTypeSet ->
                    "LOCAL_MOVE_SEARCH_QUERY"

                "DETOUR_ROUTE_COMPARISON_PACKET" in projectedTypeSet ->
                    "ROUTE_COMPARISON_QUERY"

                else ->
                    activeDetourQuestionClassFromReplyRequestV1(replyRequest)
            }

        val canonicalPrimaryPacketTypes =
            when (authoritativeDetourQuestionClass?.trim()?.uppercase()) {
                "PROOF_CHALLENGE" ->
                    listOf("DETOUR_MOVE_PROOF_PACKET")

                "TARGET_CELL_QUERY" ->
                    listOf("DETOUR_MOVE_PROOF_PACKET")

                "NEIGHBOR_CELL_QUERY" ->
                    listOf("DETOUR_LOCAL_GRID_INSPECTION_PACKET")

                "USER_REASONING_CHECK" ->
                    listOf("DETOUR_USER_PROPOSAL_VERDICT_PACKET")

                "ALTERNATIVE_TECHNIQUE_QUERY" ->
                    listOf("DETOUR_ALTERNATIVE_TECHNIQUE_PACKET")

                "LOCAL_MOVE_SEARCH_QUERY" ->
                    listOf("DETOUR_LOCAL_MOVE_SEARCH_PACKET")

                "ROUTE_COMPARISON_QUERY" ->
                    listOf("DETOUR_ROUTE_COMPARISON_PACKET")

                else ->
                    emptyList()
            }

        val projectedPrimaryPacketTypes =
            canonicalPrimaryPacketTypes.filter { it in projectedTypeSet }

        val projectedSupportPacketTypes =
            projectedTypeNames.filter { typeName ->
                typeName !in canonicalPrimaryPacketTypes &&
                        typeName != "STYLE_MINI" &&
                        typeName != "CONTINUITY_SHORT" &&
                        !(
                                authoritativeDetourQuestionClass?.trim()?.uppercase() in setOf("PROOF_CHALLENGE", "TARGET_CELL_QUERY") &&
                                        typeName == "DETOUR_LOCAL_GRID_INSPECTION_PACKET"
                                )
            }

        if (projectedPrimaryPacketTypes.isNotEmpty()) {
            out.put(JSONObject().apply {
                put("detour_mode", "packet_primary")
                put("detour_channel_family", "detour_first")
                put("detour_question_class", authoritativeDetourQuestionClass ?: JSONObject.NULL)
                put(
                    "canonical_primary_packet_types",
                    JSONArray(canonicalPrimaryPacketTypes)
                )
                put(
                    "projected_primary_packet_types",
                    JSONArray(projectedPrimaryPacketTypes)
                )
                put(
                    "projected_support_packet_types",
                    JSONArray(projectedSupportPacketTypes)
                )
                put("typed_packet_constitution_enforced", true)
                put("generic_detour_context_primary_forbidden", true)
                put("continuity_subordinate", true)
                put("local_answer_priority_required", true)
            })

            val filteredProjectedFacts = JSONArray().apply {
                for (i in 0 until projectedFacts.length()) {
                    val item = projectedFacts.optJSONObject(i) ?: continue
                    val typeName = item.optString("type", "").trim()
                    val suppressLocalInspection =
                        authoritativeDetourQuestionClass?.trim()?.uppercase() in setOf("PROOF_CHALLENGE", "TARGET_CELL_QUERY") &&
                                typeName == "DETOUR_LOCAL_GRID_INSPECTION_PACKET"

                    if (!suppressLocalInspection) put(item)
                }
            }

            for (i in 0 until filteredProjectedFacts.length()) {
                out.put(filteredProjectedFacts.getJSONObject(i))
            }

            out.put(JSONObject().apply {
                put("semantic_summary", "Canonical detour packet supplied from final projected detour facts.")
                put("field_receipt", JSONObject().apply {
                    put("pending_after", turn.pendingAfter ?: JSONObject.NULL)
                    put("story_stage_when_paused", turn.story?.stage ?: JSONObject.NULL)
                    put("return_to_route_available", allowsReturnToRoute)
                })
            })

            return out
        }

        if (canonicalPrimaryPacketTypes.isNotEmpty()) {
            out.put(JSONObject().apply {
                put("detour_mode", "missing_primary_packet_degraded")
                put("detour_channel_family", "detour_first")
                put("detour_question_class", authoritativeDetourQuestionClass ?: JSONObject.NULL)
                put(
                    "canonical_primary_packet_types",
                    JSONArray(canonicalPrimaryPacketTypes)
                )
                put(
                    "missing_primary_packet_types",
                    JSONArray(canonicalPrimaryPacketTypes)
                )
                put(
                    "available_detour_packet_types",
                    JSONArray(projectedTypeNames)
                )
                put("typed_packet_constitution_enforced", true)
                put("generic_detour_context_primary_forbidden", true)
                put("degraded_mode_requires_explicit_receipt", true)
            })

            val filteredProjectedFacts = JSONArray().apply {
                for (i in 0 until projectedFacts.length()) {
                    val item = projectedFacts.optJSONObject(i) ?: continue
                    val typeName = item.optString("type", "").trim()
                    val suppressLocalInspection =
                        authoritativeDetourQuestionClass?.trim()?.uppercase() in setOf("PROOF_CHALLENGE", "TARGET_CELL_QUERY") &&
                                typeName == "DETOUR_LOCAL_GRID_INSPECTION_PACKET"

                    if (!suppressLocalInspection) put(item)
                }
            }

            for (i in 0 until filteredProjectedFacts.length()) {
                out.put(filteredProjectedFacts.getJSONObject(i))
            }

            out.put(JSONObject().apply {
                put("semantic_summary", "Canonical detour packet missing after final detour projection; degraded detour facts supplied.")
                put("field_receipt", JSONObject().apply {
                    put("pending_after", turn.pendingAfter ?: JSONObject.NULL)
                    put("story_stage_when_paused", turn.story?.stage ?: JSONObject.NULL)
                    put("return_to_route_available", allowsReturnToRoute)
                })
            })

            return out
        }

        if (projectedTypeNames.isNotEmpty()) {
            out.put(JSONObject().apply {
                put("detour_mode", "packet_fallback_no_canonical_primary")
                put("detour_channel_family", "detour_first")
                put("detour_question_class", authoritativeDetourQuestionClass ?: JSONObject.NULL)
                put("typed_packet_constitution_enforced", false)
                put("continuity_subordinate", true)
                put("local_answer_priority_required", true)
            })

            for (i in 0 until projectedFacts.length()) {
                out.put(projectedFacts.getJSONObject(i))
            }

            out.put(JSONObject().apply {
                put("semantic_summary", "Detour projected facts supplied without canonical primary packet classification.")
                put("field_receipt", JSONObject().apply {
                    put("pending_after", turn.pendingAfter ?: JSONObject.NULL)
                    put("story_stage_when_paused", turn.story?.stage ?: JSONObject.NULL)
                    put("return_to_route_available", allowsReturnToRoute)
                })
            })

            return out
        }

        return bridgeFacts
    }



    private fun shouldIncludeSolvingContinuityShort(replyRequest: ReplyRequestV1): Boolean {
        val routeReturnAllowed = replyRequest.turn.turnRouteReturnAllowed
        val routeReturn =
            if (routeReturnAllowed) projectDetourRouteReturnPacket(replyRequest) else JSONObject()
        if (routeReturnAllowed && routeReturn.length() > 0) return true

        val handover = projectHandoverNoteMini(replyRequest)
        if (handover.length() > 0) return true

        val pendingBefore = replyRequest.turn.pendingBefore.orEmpty()
        val pendingAfter = replyRequest.turn.pendingAfter.orEmpty()

        return pendingBefore.contains("UserAgendaBridge", ignoreCase = true) ||
                pendingAfter.contains("UserAgendaBridge", ignoreCase = true) ||
                (routeReturnAllowed && pendingBefore.contains("Return", ignoreCase = true)) ||
                (routeReturnAllowed && pendingAfter.contains("Return", ignoreCase = true))
    }

    fun projectFactsForSetupDemand(replyRequest: ReplyRequestV1): JSONArray {
        if (isUserAgendaBridgeTurnV1(replyRequest)) {
            return if (hasAnyDetourPacketV1(replyRequest)) {
                buildDetourOnlyFactsV1(replyRequest)
            } else {
                buildUserAgendaBridgeFactsV1(replyRequest)
            }
        }

        if (hasAnyDetourPacketV1(replyRequest)) {
            return buildDetourOnlyFactsV1(replyRequest)
        }

        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", projectCtaContext(replyRequest)) })

            put(JSONObject().apply { put("type", "SETUP_REPLY_PACKET"); put("payload", projectSetupReplyPacket(replyRequest)) })

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0 && shouldIncludeSolvingContinuityShort(replyRequest)) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }
        }
    }

    fun projectFactsForConfrontationDemand(replyRequest: ReplyRequestV1): JSONArray {
        if (isUserAgendaBridgeTurnV1(replyRequest)) {
            return if (hasAnyDetourPacketV1(replyRequest)) {
                buildDetourOnlyFactsV1(replyRequest)
            } else {
                buildUserAgendaBridgeFactsV1(replyRequest)
            }
        }

        if (hasAnyDetourPacketV1(replyRequest)) {
            return buildDetourOnlyFactsV1(replyRequest)
        }

        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", projectCtaContext(replyRequest)) })

            put(JSONObject().apply { put("type", "CONFRONTATION_REPLY_PACKET"); put("payload", projectConfrontationReplyPacket(replyRequest)) })

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0 && shouldIncludeSolvingContinuityShort(replyRequest)) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }
        }
    }

    fun projectFactsForResolutionDemand(replyRequest: ReplyRequestV1): JSONArray {
        if (isUserAgendaBridgeTurnV1(replyRequest)) {
            return if (hasAnyDetourPacketV1(replyRequest)) {
                buildDetourOnlyFactsV1(replyRequest)
            } else {
                buildUserAgendaBridgeFactsV1(replyRequest)
            }
        }

        if (hasAnyDetourPacketV1(replyRequest)) {
            return buildDetourOnlyFactsV1(replyRequest)
        }

        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", projectCtaContext(replyRequest)) })

            put(JSONObject().apply { put("type", "RESOLUTION_REPLY_PACKET"); put("payload", projectResolutionReplyPacket(replyRequest)) })

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0 && shouldIncludeSolvingContinuityShort(replyRequest)) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }
        }
    }

    fun projectFactsForRepairDemand(replyRequest: ReplyRequestV1): JSONArray {
        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", projectContinuityShort(replyRequest)) })
            put(JSONObject().apply { put("type", "REPAIR_CONTEXT"); put("payload", projectRepairContext(replyRequest)) })

            val routeReturnAllowed = replyRequest.turn.turnRouteReturnAllowed
            if (routeReturnAllowed) {
                val routeReturn = projectDetourRouteReturnPacket(replyRequest)
                if (routeReturn.length() > 0) {
                    put(JSONObject().apply {
                        put("type", "DETOUR_ROUTE_RETURN_PACKET")
                        put("payload", routeReturn)
                    })
                }
            }

            val cta = projectCtaContext(replyRequest)
            if (cta.length() > 0) put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", cta) })
        }
    }

    fun projectFactsForRouteJumpDemand(replyRequest: ReplyRequestV1): JSONArray {
        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }

            val cta = projectCtaContext(replyRequest)
            if (cta.length() > 0) {
                put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", cta) })
            }

            when (replyRequest.turn.story?.stage) {
                "SETUP" -> put(JSONObject().apply {
                    put("type", "SETUP_REPLY_PACKET")
                    put("payload", projectSetupReplyPacket(replyRequest))
                })

                "CONFRONTATION" -> put(JSONObject().apply {
                    put("type", "CONFRONTATION_REPLY_PACKET")
                    put("payload", projectConfrontationReplyPacket(replyRequest))
                })

                "RESOLUTION" -> put(JSONObject().apply {
                    put("type", "RESOLUTION_REPLY_PACKET")
                    put("payload", projectResolutionReplyPacket(replyRequest))
                })
            }
        }
    }

    // ---------------------------------------------------------------------
    // Channel dispatcher
    // ---------------------------------------------------------------------

    fun projectChannel(
        replyRequest: ReplyRequestV1,
        channel: ReplySupplyChannelV1
    ): JSONObject {
        return when (channel) {
            ReplySupplyChannelV1.TURN_HEADER_MINI -> projectTurnHeaderMini(replyRequest)
            ReplySupplyChannelV1.STYLE_MINI -> projectStyleMini(replyRequest)
            ReplySupplyChannelV1.DECISION_SUMMARY_MINI -> projectDecisionSummaryMini(replyRequest)
            ReplySupplyChannelV1.CONTINUITY_SHORT -> projectContinuityShort(replyRequest)
            ReplySupplyChannelV1.PERSONALIZATION_MINI -> projectPersonalizationMini(replyRequest)
            ReplySupplyChannelV1.CTA_CONTEXT -> projectCtaContext(replyRequest)
            ReplySupplyChannelV1.SOLVABILITY_CONTEXT -> projectSolvabilityContext(replyRequest)

            ReplySupplyChannelV1.ONBOARDING_CONTEXT -> projectOnboardingContext(replyRequest)

            ReplySupplyChannelV1.CONFIRMING_CONTEXT -> projectConfirmingContext(replyRequest)
            ReplySupplyChannelV1.PENDING_CONTEXT_CHANNEL -> projectPendingContextChannel(replyRequest)
            ReplySupplyChannelV1.GRID_VALIDATION_CONTEXT -> projectGridValidationContext(replyRequest)
            ReplySupplyChannelV1.GRID_CANDIDATE_CONTEXT -> projectGridCandidateContext(replyRequest)


            ReplySupplyChannelV1.GRID_OCR_TRUST_CONTEXT -> projectGridOcrTrustContext(replyRequest)
            ReplySupplyChannelV1.GRID_CONTENTS_CONTEXT -> projectGridContentsContext(replyRequest)
            ReplySupplyChannelV1.GRID_CHANGELOG_CONTEXT -> projectGridChangelogContext(replyRequest)

            ReplySupplyChannelV1.GRID_MUTATION_CONTEXT -> projectGridMutationContext(replyRequest)
            ReplySupplyChannelV1.SOLVING_SUPPORT_CONTEXT -> projectSolvingSupportContext(replyRequest)

            ReplySupplyChannelV1.DETOUR_CONTEXT -> projectDetourContext(replyRequest)
            ReplySupplyChannelV1.DETOUR_MOVE_PROOF_PACKET -> projectDetourMoveProofPacket(replyRequest)
            ReplySupplyChannelV1.DETOUR_LOCAL_GRID_INSPECTION_PACKET -> projectDetourLocalGridInspectionPacket(replyRequest)
            ReplySupplyChannelV1.DETOUR_USER_PROPOSAL_VERDICT_PACKET -> projectDetourUserProposalVerdictPacket(replyRequest)
            ReplySupplyChannelV1.DETOUR_ALTERNATIVE_TECHNIQUE_PACKET -> projectDetourAlternativeTechniquePacket(replyRequest)
            ReplySupplyChannelV1.DETOUR_LOCAL_MOVE_SEARCH_PACKET -> projectDetourLocalMoveSearchPacket(replyRequest)
            ReplySupplyChannelV1.DETOUR_ROUTE_COMPARISON_PACKET -> projectDetourRouteComparisonPacket(replyRequest)
            ReplySupplyChannelV1.DETOUR_NARRATIVE_CONTEXT -> projectDetourNarrativeContext(replyRequest)
            ReplySupplyChannelV1.PREFERENCE_CONTEXT -> projectPreferenceContext(replyRequest)

            ReplySupplyChannelV1.META_CONTEXT -> projectMetaContext(replyRequest)
            ReplySupplyChannelV1.HELP_CONTEXT -> projectHelpContext(replyRequest)
            ReplySupplyChannelV1.FREE_TALK_CONTEXT -> projectFreeTalkContext(replyRequest)


            ReplySupplyChannelV1.SETUP_REPLY_PACKET -> projectSetupReplyPacket(replyRequest)

            ReplySupplyChannelV1.SETUP_STORY_SLICE -> projectSetupStorySlice(replyRequest)
            ReplySupplyChannelV1.SETUP_STEP_SLICE -> projectSetupStepSlice(replyRequest)

            ReplySupplyChannelV1.CONFRONTATION_REPLY_PACKET -> projectConfrontationReplyPacket(replyRequest)
            ReplySupplyChannelV1.CONFRONTATION_STORY_SLICE -> projectConfrontationStorySlice(replyRequest)
            ReplySupplyChannelV1.CONFRONTATION_STEP_SLICE -> projectConfrontationStepSlice(replyRequest)
            ReplySupplyChannelV1.RESOLUTION_REPLY_PACKET -> projectResolutionReplyPacket(replyRequest)
            ReplySupplyChannelV1.RESOLUTION_STORY_SLICE -> projectResolutionStorySlice(replyRequest)

            ReplySupplyChannelV1.RESOLUTION_STEP_SLICE -> projectResolutionStepSlice(replyRequest)
            ReplySupplyChannelV1.GLOSSARY_MINI -> projectGlossaryMini(replyRequest)
            ReplySupplyChannelV1.TECHNIQUE_CARD_MINI -> projectTechniqueCardMini(replyRequest)
            ReplySupplyChannelV1.HANDOVER_NOTE_MINI -> projectHandoverNoteMini(replyRequest)
            ReplySupplyChannelV1.OVERLAY_MINI -> projectOverlayMini(replyRequest)
            ReplySupplyChannelV1.REPAIR_CONTEXT -> projectRepairContext(replyRequest)
        }
    }



    fun projectFactsForOnboardingDemand(replyRequest: ReplyRequestV1): JSONArray {
        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "DECISION_SUMMARY_MINI"); put("payload", projectDecisionSummaryMini(replyRequest)) })
            put(JSONObject().apply { put("type", "ONBOARDING_CONTEXT"); put("payload", projectOnboardingContext(replyRequest)) })

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }
        }
    }

    fun projectFactsForConfirmingDemand(replyRequest: ReplyRequestV1): JSONArray {
        if (isUserAgendaBridgeTurnV1(replyRequest)) {
            return if (hasAnyDetourPacketV1(replyRequest)) {
                buildDetourOnlyFactsV1(replyRequest)
            } else {
                buildUserAgendaBridgeFactsV1(replyRequest)
            }
        }

        return JSONArray().apply {
            put(JSONObject().apply { put("type", "TURN_HEADER_MINI"); put("payload", projectTurnHeaderMini(replyRequest)) })
            put(JSONObject().apply { put("type", "STYLE_MINI"); put("payload", projectStyleMini(replyRequest)) })
            put(JSONObject().apply { put("type", "DECISION_SUMMARY_MINI"); put("payload", projectDecisionSummaryMini(replyRequest)) })
            put(JSONObject().apply { put("type", "CTA_CONTEXT"); put("payload", projectCtaContext(replyRequest)) })
            put(JSONObject().apply { put("type", "SOLVABILITY_CONTEXT"); put("payload", projectSolvabilityContext(replyRequest)) })
            put(JSONObject().apply { put("type", "CONFIRMING_CONTEXT"); put("payload", projectConfirmingContext(replyRequest)) })

            val continuity = projectContinuityShort(replyRequest)
            if (continuity.length() > 0) {
                put(JSONObject().apply { put("type", "CONTINUITY_SHORT"); put("payload", continuity) })
            }

            val glossary = projectGlossaryMini(replyRequest)
            if (glossary.length() > 0) {
                put(JSONObject().apply { put("type", "GLOSSARY_MINI"); put("payload", glossary) })
            }
        }
    }
}